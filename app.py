from flask import Flask, render_template, request, session, redirect, url_for
import joblib, numpy as np, os, sys, warnings, datetime
warnings.filterwarnings("ignore")
import pandas as pd

sys.path.insert(0, ".")
from src.features import extract_features, FEATURE_NAMES

app = Flask(__name__)
app.secret_key = "phishnet-swce-2026"
BASE = os.path.dirname(os.path.abspath(__file__))

# ── TRUSTED DOMAIN WHITELIST ──────────────────────────────────
# These are globally recognised legitimate domains.
# Real security tools (Google Safe Browsing, Kaspersky) all maintain
# a trusted whitelist — this is standard industry practice.
TRUSTED_DOMAINS = {
    # Search engines
    "google.com","www.google.com","google.co.in","www.google.co.in",
    "bing.com","www.bing.com","yahoo.com","www.yahoo.com",
    "duckduckgo.com","www.duckduckgo.com",
    # Social media
    "facebook.com","www.facebook.com","instagram.com","www.instagram.com",
    "twitter.com","www.twitter.com","x.com","www.x.com",
    "linkedin.com","www.linkedin.com","reddit.com","www.reddit.com",
    "pinterest.com","www.pinterest.com","tiktok.com","www.tiktok.com",
    "whatsapp.com","www.whatsapp.com","telegram.org","www.telegram.org",
    # Shopping
    "amazon.com","www.amazon.com","amazon.in","www.amazon.in",
    "flipkart.com","www.flipkart.com","ebay.com","www.ebay.com",
    "walmart.com","www.walmart.com","etsy.com","www.etsy.com",
    # Tech / Cloud
    "microsoft.com","www.microsoft.com","apple.com","www.apple.com",
    "github.com","www.github.com","stackoverflow.com","www.stackoverflow.com",
    "openai.com","www.openai.com","chatgpt.com","www.chatgpt.com",
    "claude.ai","anthropic.com","gemini.google.com",
    # Video / Entertainment
    "youtube.com","www.youtube.com","netflix.com","www.netflix.com",
    "spotify.com","www.spotify.com","twitch.tv","www.twitch.tv",
    "imdb.com","www.imdb.com","hotstar.com","www.hotstar.com",
    # News
    "bbc.com","www.bbc.com","cnn.com","www.cnn.com",
    "nytimes.com","www.nytimes.com","thehindu.com","www.thehindu.com",
    "ndtv.com","www.ndtv.com","timesofindia.com","www.timesofindia.com",
    # Education
    "wikipedia.org","www.wikipedia.org","coursera.org","www.coursera.org",
    "udemy.com","www.udemy.com","khanacademy.org","www.khanacademy.org",
    "nptel.ac.in","www.nptel.ac.in","edx.org","www.edx.org",
    # Finance
    "paypal.com","www.paypal.com","razorpay.com","www.razorpay.com",
    "paytm.com","www.paytm.com","phonepe.com","www.phonepe.com",
    # Government / Edu
    "gov.in","nic.in","ac.in","edu","mit.edu","stanford.edu",
    # Cloud / Productivity
    "drive.google.com","docs.google.com","mail.google.com","meet.google.com",
    "outlook.com","www.outlook.com","office.com","www.office.com",
    "dropbox.com","www.dropbox.com","notion.so","www.notion.so",
    # Developer tools
    "kaggle.com","www.kaggle.com","colab.research.google.com",
    "replit.com","www.replit.com","heroku.com","www.heroku.com",
    "render.com","www.render.com","vercel.com","www.vercel.com",
    "netlify.com","www.netlify.com","aws.amazon.com","azure.microsoft.com",
    "cloud.google.com","digitalocean.com","www.digitalocean.com",
}

def _is_trusted(url: str) -> bool:
    """Check if URL belongs to a globally trusted domain."""
    try:
        from urllib.parse import urlparse
        host = urlparse(url).netloc.lower().split(":")[0]
        # Direct match
        if host in TRUSTED_DOMAINS:
            return True
        # Check if it ends with a trusted domain
        # e.g. maps.google.com should be trusted because google.com is trusted
        for td in TRUSTED_DOMAINS:
            if host == td or host.endswith("." + td):
                if len(td.split(".")) >= 2:  # skip single-word entries
                    return True
        return False
    except:
        return False

def _load():
    try:
        rf  = joblib.load(os.path.join(BASE, "model", "rf_model.pkl"))
        gb  = joblib.load(os.path.join(BASE, "model", "gb_model.pkl"))
        sc  = joblib.load(os.path.join(BASE, "model", "scaler.pkl"))
        fn  = joblib.load(os.path.join(BASE, "model", "feature_names.pkl"))
        print("✅ Models loaded successfully")
        return rf, gb, sc, fn
    except Exception as e:
        print(f"❌ Model load error: {e}")
        print("   Run:  python train.py  first!")
        return None, None, None, None

RF, GB, SCALER, FEAT_NAMES = _load()

def _predict(url: str, model_name: str) -> dict:
    model = GB if "gradient" in model_name.lower() else RF
    if model is None or SCALER is None:
        raise RuntimeError("Model not loaded. Run python train.py first.")
    threat = "LOW"   # 🔥 default value (IMPORTANT FIX)

    # ── WHITELIST CHECK FIRST ─────────────────────────────────
    if _is_trusted(url):
        return {
            "url":          url,
            "is_malicious": False,
            "status":       "SAFE — Legitimate Website",
            "confidence":   99.0,
            "mal_prob":     0.5,
            "model":        model_name,
            "color":        "green",
            "emoji":        "✅",
            "note":         "Verified trusted domain",
            "time":         datetime.datetime.now().strftime("%d-%m-%Y  %H:%M:%S"),
            "threat": "LOW",
            "reasons": ["Trusted domain (whitelisted)"],     
        }

    # ── ML PREDICTION ─────────────────────────────────────────
    feats    = extract_features(url)
    col_names = FEAT_NAMES if FEAT_NAMES else FEATURE_NAMES
    df_input  = pd.DataFrame([feats], columns=col_names)
    scaled    = SCALER.transform(df_input)
    pred      = int(model.predict(scaled)[0])
    prob      = model.predict_proba(scaled)[0]

    is_malicious = (pred == 1)
    conf         = round(float(max(prob)) * 100, 1)
    mal_prob     = round(float(prob[1]) * 100, 1) if len(prob) > 1 else (conf if is_malicious else round(100-conf,1))
        # 🔥 ADDED: Explanation system (SAFE ADDITION)
    reasons = []

    # Shortened URL detection
    is_suspicious = False   # 🔥 ADD THIS ABOVE (only once)

    if any(short in url for short in ["bit.ly", "tinyurl", "goo.gl", "t.co"]):
        reasons.append("Shortened URL detected (may hide destination)")
    is_suspicious = True   # 🔥 ADD THIS LINE
    # Suspicious patterns
    if "@" in url:
        reasons.append("Contains @ symbol (URL redirection trick)")

    if "-" in url:
        reasons.append("Hyphen in domain (common in phishing)")

    if len(url) > 75:
        reasons.append("Unusually long URL")

    if url.startswith("http://"):
        reasons.append("Not secure (HTTP)")

    # 🔥 Threat level
    if is_malicious:
        threat = "HIGH" if conf > 85 else "MEDIUM"
    else:
        threat = "LOW" if conf > 80 else "MEDIUM"

    return {
        "url":          url,
        "is_malicious": is_malicious,
        "status":       "UNSAFE — Malicious / Phishing" if is_malicious else "SAFE — Legitimate",
        "confidence":   conf,
        "mal_prob":     mal_prob,
        "model":        model_name,
        "color":        "red" if is_malicious else "green",
        "emoji":        "🚨" if is_malicious else "✅",
        "note":         "",
        "time":         datetime.datetime.now().strftime("%d-%m-%Y  %H:%M:%S"),
        "threat":        threat,
        "reasons":       reasons,
        "is_suspicious": is_suspicious,
    }

# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if (request.form.get("username") == "admin" and
                request.form.get("password") == "admin"):
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        error = "Incorrect credentials. Use  admin / admin"
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/dashboard")
def dashboard():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("dashboard.html")

@app.route("/scanner", methods=["GET", "POST"])
def scanner():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    result = None
    if request.method == "POST":
        url   = request.form.get("url", "").strip()
        mname = request.form.get("model", "Random Forest")
        if url:
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            try:
                result = _predict(url, mname)
            except Exception as e:
                result = {
                    "error": str(e),
                    "url":   url,
                    "model": mname,
                    "time":  datetime.datetime.now().strftime("%d-%m-%Y  %H:%M:%S"),
                }
    return render_template("scanner.html", result=result)

@app.route("/performance")
def performance():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("performance.html")

@app.route("/charts")
def charts():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("charts.html")

@app.route("/faq")
def faq():
    return render_template("faq.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
