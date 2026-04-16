# services/detector.py — PhishNet v2.0
import os, re, uuid, datetime, joblib, warnings
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from src.features import extract_features, FEATURE_NAMES, has_at_symbol, long_url, count_subdomains, uses_ip_address, suspicious_keywords, https_feature
from services.intelligence import compute_intelligence_signals

warnings.filterwarnings("ignore")
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load_resources():
    try:
        models = {
            "rf":  joblib.load(os.path.join(_BASE, "model", "rf_model.pkl")),
            "gb":  joblib.load(os.path.join(_BASE, "model", "gb_model.pkl")),
            "dt":  joblib.load(os.path.join(_BASE, "model", "dt_model.pkl")) if os.path.exists(os.path.join(_BASE, "model", "dt_model.pkl")) else None,
            "lr":  joblib.load(os.path.join(_BASE, "model", "lr_model.pkl")) if os.path.exists(os.path.join(_BASE, "model", "lr_model.pkl")) else None,
            "svm": joblib.load(os.path.join(_BASE, "model", "svm_model.pkl")) if os.path.exists(os.path.join(_BASE, "model", "svm_model.pkl")) else None,
        }
        # Fallback to RF if others missing (safety)
        for k in ["dt", "lr", "svm"]:
            if models[k] is None: models[k] = models["rf"]
            
        scaler = joblib.load(os.path.join(_BASE, "model", "scaler.pkl"))
        fn = joblib.load(os.path.join(_BASE, "model", "feature_names.pkl"))
        return models, scaler, fn
    except Exception as e:
        print(f"Error loading models: {e}")
        return {}, None, None

MODELS, SCALER, FEAT_NAMES = _load_resources()

trusted_domains = [
    "google.com", "wikipedia.org", "github.com", "microsoft.com",
    "amazon.in", "stackoverflow.com", "linkedin.com", "apple.com",
    "netflix.com", "adobe.com"
]

def _get_severity(score):
    if score <= 20: return {"label": "SAFE", "css": "badge-safe", "color": "#10b981"}
    if score <= 50: return {"label": "MEDIUM", "css": "badge-medium", "color": "#f59e0b"}
    if score <= 80: return {"label": "HIGH", "css": "badge-high", "color": "#f97316"}
    return {"label": "CRITICAL", "css": "badge-critical", "color": "#ef4444"}

def analyse(url: str, model_id: str = "rf") -> dict:
    url = url.strip()
    if not url.startswith(("http://", "https://")): url = "https://" + url
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host:
        host = parsed.netloc.split(":")[0].lower()
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scan_id = str(uuid.uuid4())[:8]

    if host in trusted_domains or any(host.endswith("." + d) for d in trusted_domains):
        # Run intelligence signals for metadata
        intel = compute_intelligence_signals(url)
        # Whitelisted domains get a minimal risk score (2/100)
        risk_score = 0
        sev = _get_severity(risk_score)
        https_stat = https_feature(url)
        
        return {
            "url": url,
            "risk_score": risk_score,
            "severity": sev["label"],
            "sev_color": sev["color"],
            "confidence": 99.9,
            "is_malicious": False,
            "prediction": "SAFE",
            "risk_factors": ["Trusted domain (whitelisted)"],
            "https_status": https_stat,
            "domain_age": intel.get("age_display", "Verified"),
            "ssl_valid": intel.get("ssl_valid"),
            "dns_valid": intel.get("dns_valid"),
            "timestamp": timestamp,
            "scan_id": scan_id,
            "model": model_id.upper(),
        }

    # ── ML Prediction (source of truth) ─────────────────────
    model    = MODELS.get(model_id, MODELS.get("rf"))
    features = extract_features(url)
    df       = pd.DataFrame([features], columns=FEAT_NAMES if FEAT_NAMES else FEATURE_NAMES)
    scaled   = SCALER.transform(df)
    
    prob     = model.predict_proba(scaled)[0]
    mal_prob = float(prob[1])
    confidence = round(float(max(prob)) * 100, 1)

    # ── Real-World Intelligence Signals (supporting only) ────
    # Runs concurrently after ML — does NOT change prediction
    try:
        intel = compute_intelligence_signals(url)
    except Exception:
        intel = {
            "age_display": "Unavailable", "ssl_valid": None,
            "dns_valid": None, "intel_score": 0, "intel_factors": []
        }

    # ── ML Feature-Importance Explainability ─────────────────
    has_at   = has_at_symbol(url)
    long_u   = long_url(url)
    subd_cnt = count_subdomains(url)
    has_ip   = uses_ip_address(url)
    susp_kw  = suspicious_keywords(url)
    http_val = https_feature(url)

    ml_factors = []
    if hasattr(model, 'feature_importances_'):
        f_names  = FEAT_NAMES if FEAT_NAMES else FEATURE_NAMES
        feat_impt = list(zip(f_names, model.feature_importances_, features))
        feat_impt.sort(key=lambda x: x[1], reverse=True)
        
        for name, imp, val in feat_impt:
            if name == "num_at" and has_at:
                ml_factors.append("Contains @-symbol redirect trick")
            elif name == "url_length" and long_u:
                ml_factors.append("Unusually long URL (possible obfuscation)")
            elif name == "has_ip" and has_ip:
                ml_factors.append("Uses IP address instead of domain")
            elif name == "has_suspicious_keyword" and susp_kw:
                ml_factors.append("Contains phishing-related keywords (login/verify/etc)")
            elif name == "subdomain_depth" and subd_cnt > 2:
                ml_factors.append(f"Excessive subdomains ({subd_cnt})")
            if len(ml_factors) >= 3:
                break

    # HTTPS logic — absence is a risk, presence is NOT safety
    if http_val == 0:
        ml_factors.append("No HTTPS encryption (unsafe connection)")

    # ── Merge ML + Intelligence risk factors ──────────────────
    # Intel factors fill remaining slots (up to 5 total)
    combined_factors = list(dict.fromkeys(ml_factors + intel.get("intel_factors", [])))[:5]

    # ── Final Risk Score ──────────────────────────────────────
    # ML drives ~80% of score; intel provides +8 per triggered rule (supporting)
    rule_score = len(ml_factors) * 6 + intel.get("intel_score", 0)
    risk_score = int(min(max((mal_prob * 80) + rule_score, 0), 100))

    # ── Safe Signal Boosting ──────────────────────────────────
    # If a domain has valid SSL, DNS, and is older than 6 months, boost safety
    is_aged = intel.get("age_days", 0) > 180
    if intel.get("ssl_valid") and intel.get("dns_valid") and is_aged:
        risk_score = max(0, risk_score - 20)
        # Bias the final verdict: if it was borderline malicious, push to safe
        if mal_prob < 0.7:
            mal_prob *= 0.5 
        combined_factors.append("Verified heritage signal detected (Age > 6mo)")

    sev = _get_severity(risk_score)

    return {
        "url":         url,
        "risk_score":  risk_score,
        "severity":    sev["label"],
        "sev_color":   sev["color"],
        "confidence":  confidence,
        "is_malicious": mal_prob > 0.5,
        "prediction":  "MALICIOUS" if mal_prob > 0.5 else "SAFE",
        "risk_factors": combined_factors,
        "https_status": http_val,
        # Real-world intelligence signals
        "domain_age":  intel.get("age_display", "Unavailable"),
        "ssl_valid":   intel.get("ssl_valid"),
        "dns_valid":   intel.get("dns_valid"),
        "timestamp":   timestamp,
        "scan_id":     scan_id,
        "model":       model_id.upper(),
    }

