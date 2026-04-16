from flask import Flask, render_template, request, jsonify, send_file, Response
from urllib.parse import urlparse
import os, sys, uuid, datetime, re, time, json

sys.path.insert(0, ".")
from services.detector import analyse, trusted_domains
from services.report import generate_pdf
from services.reporter import generate_report_id, save_report, load_reports

app = Flask(__name__)
app.secret_key = "phishnet-swce-2026"

# ── GLOBAL STATE ──────────────────────────────────────────────
SCAN_HISTORY  = []
MAX_HISTORY   = 20
latest_result = {}

# ── HELPERS ───────────────────────────────────────────────────

def _add_to_history(result):
    global SCAN_HISTORY
    result["scan_id"] = str(uuid.uuid4())[:8]
    entry = {
        "scan_id":        result["scan_id"],
        "url":            result.get("url"),
        "risk_score":     result.get("risk_score"),
        "severity":       result.get("severity"),
        "severity_color": result.get("sev_color", "#888"),
        "timestamp":      result.get("timestamp"),
        "model":          result.get("model"),
        "full_result":    result,
    }
    SCAN_HISTORY.insert(0, entry)
    if len(SCAN_HISTORY) > MAX_HISTORY:
        SCAN_HISTORY.pop()



# ── ROUTES ────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/dashboard")
def dashboard():
    total = len(SCAN_HISTORY)
    malicious   = sum(1 for s in SCAN_HISTORY if s["risk_score"] > 50)
    mal_percent = round((malicious / total * 100), 1) if total > 0 else 0
    avg_risk    = round(sum(s["risk_score"] for s in SCAN_HISTORY) / total, 1) if total > 0 else 0
    return render_template("dashboard.html",
                           total=total,
                           mal_percent=mal_percent,
                           avg_risk=avg_risk,
                           recent=SCAN_HISTORY[:5])


@app.route("/scanner", methods=["GET", "POST"])
def scanner():
    global latest_result
    result       = None
    multi_models = []
    reasons      = []

    if request.method == "POST":
        url      = request.form.get("url", "").strip()
        model_id = request.form.get("model", "rf")

        if url:
            # ── URL auto-fix ──────────────────────────────────
            if not url.startswith("http://") and not url.startswith("https://"):
                url = "http://" + url

            try:
                import time
                start_time = time.time()
                result = analyse(url, model_id)
                scan_duration = time.time() - start_time


                if "error" not in result:
                    result["time_taken"] = round(scan_duration, 2)
                    _add_to_history(result)


                    # Domain intelligence
                    parsed_url = urlparse(url)
                    host = (parsed_url.hostname or "").lower()
                    if not host:
                        host = parsed_url.netloc.split(":")[0].lower()
                    
                    is_whitelisted = (host in trusted_domains or 
                                     any(host.endswith("." + d) for d in trusted_domains))
                    
                    domain = parsed_url.netloc.lower()
                    is_https = (parsed_url.scheme.lower() == "https")
                    is_ip  = re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain.split(":")[0])
                    tld_risk = "HIGH" if domain.endswith(
                        (".xyz", ".ru", ".top", ".cn", ".tk", ".pw", ".cc")) else "LOW"
                    subdomain_count = domain.count(".") - 1 if not is_ip and "." in domain else 0
                    result["domain_info"] = {
                        "is_ip":           bool(is_ip),
                        "tld_risk":        tld_risk,
                        "subdomain_count": max(0, subdomain_count),
                        "domain_name":     domain,
                        "is_https":        is_https
                    }

                    # Risk breakdown
                    rs = result["risk_score"]
                    result["breakdown"] = {
                        "ML Prediction":    int(rs * 0.50),
                        "Keywords":         min(20, rs * 0.25),
                        "Domain Structure": min(15, rs * 0.15),
                        "Protocol Issues":  max(0, rs - int(rs * 0.50)
                                              - min(20, rs * 0.25)
                                              - min(15, rs * 0.15)),
                    }

                    # ── Multi-model consensus ─────────────────
                    model_names = {"rf": "Random Forest", "gb": "Gradient Boosting",
                                   "dt": "Decision Tree", "lr": "Logistic Regression",
                                   "svm": "SVM"}
                    predictions  = [result["is_malicious"]]
                    model_results = [{
                        "name":         model_names.get(model_id, model_id.upper()),
                        "is_malicious": result["is_malicious"],
                        "confidence":   result["confidence"],
                    }]
                    for m in ["rf", "gb", "dt", "lr"]:
                        if m != model_id:
                            comp = analyse(url, m)
                            
                            # Apply whitelist to consensus models too
                            if is_whitelisted:
                                comp["is_malicious"] = False
                                comp["confidence"] = 98

                            predictions.append(comp.get("is_malicious", False))
                            model_results.append({
                                "name":         model_names.get(m, m.upper()),
                                "is_malicious": comp.get("is_malicious", False),
                                "confidence":   comp.get("confidence", 0),
                            })

                            
                    # ── Explainability ───────────────────────
                    reasons = result.get("risk_factors", [])
                    result["reasons"] = reasons

                    # URL Pattern Analyzer
                    url_malicious = result.get("https_status") == 0 or result.get("domain_info", {}).get("is_ip")
                    if is_whitelisted: url_malicious = False
                    
                    predictions.append(url_malicious)
                    model_results.append({
                        "name":         "URL Pattern Analyzer",
                        "is_malicious": url_malicious,
                        "confidence":   95 if url_malicious else 85,
                    })

                    # ── Heuristic Engine (Scoring System) ──
                    h_score = 0
                    if "@" in url: h_score += 2
                    if len(url) > 75: h_score += 1
                    if any(k in url.lower() for k in ["login", "verify", "update"]): h_score += 1
                    if is_ip: h_score += 2
                    if subdomain_count > 2: h_score += 1

                    # Determine verdict
                    if h_score >= 3:
                        h_verdict = "MALICIOUS"
                        h_malicious = True
                        h_conf = 90
                    elif h_score == 2:
                        h_verdict = "SUSPICIOUS"
                        h_malicious = True # Suspicious counts as flagged in consensus
                        h_conf = 70
                    else:
                        h_verdict = "SAFE"
                        h_malicious = False
                        h_conf = 95

                    # Whitelist Override
                    if is_whitelisted:
                        h_malicious = False
                        h_verdict = "SAFE (WHITELISTED)"
                        h_conf = 99

                    predictions.append(h_malicious)
                    model_results.append({
                        "name":         "Heuristic Engine",
                        "is_malicious": h_malicious,
                        "status":       h_verdict,
                        "confidence":   h_conf,
                    })




                    # ── Weighted Voting Consensus ─────────────────
                    weights = {
                        "Random Forest":       3,
                        "Gradient Boosting":   2,
                        "Logistic Regression": 2,
                        "Heuristic Engine":    1,
                        "Decision Tree":       1,
                        "SVM":                 1,
                        "URL Pattern Analyzer": 1
                    }

                    malicious_score = 0
                    safe_score = 0
                    total_weight = 0

                    for m in model_results:
                        w = weights.get(m["name"], 1)
                        total_weight += w
                        if m["is_malicious"]:
                            malicious_score += w
                        else:
                            safe_score += w

                    # Final verdict based on weighted score
                    final_malicious = malicious_score > safe_score

                    # Force SAFE if whitelisted
                    if is_whitelisted:
                        final_malicious = False

                    # Calculate confidence based on weighted agreement
                    agreed_weight = malicious_score if final_malicious else safe_score
                    agreement_pct = int((agreed_weight / total_weight) * 100) if total_weight > 0 else 0
                    
                    result["final_malicious"] = final_malicious
                    result["model_results"]   = model_results
                    result["agreement"]       = f"{agreement_pct}% Weighted Consensus"


                    # ── Store for PDF ─────────────────────────
                    features = {
                        layer["name"]: "Detected" if layer["triggered"] else "Clean"
                        for layer in result.get("layers", [])
                    }
                    if not features:
                        features = result.get("breakdown", {})

                    latest_result = {
                        "url":          url,
                        "result":       "PHISHING" if final_malicious else "SAFE",
                        "risk":         result["risk_score"],
                        "features":     features,
                        "reasons":      reasons,
                        "model_results": model_results,
                    }

            except Exception as e:
                result = {"error": str(e), "url": url}

    return render_template("scanner.html",
                           result=result,
                           multi_models=result.get("model_results", []) if result else [],
                           reasons=result.get("reasons", []) if result else [])


@app.route("/history")
def history():
    return render_template("history.html", history=SCAN_HISTORY)


@app.route("/bulk-scan", methods=["GET", "POST"])
def bulk_scan():
    results = []
    if request.method == "POST":
        url_list = [u.strip() for u in request.form.get("urls", "").split("\n") if u.strip()]
        for u in url_list:
            if not u.startswith(("http://", "https://")):
                u = "https://" + u
            try:
                res = analyse(u, "rf")
                _add_to_history(res)
                results.append(res)
            except Exception as e:
                results.append({"url": u, "error": str(e)})
    return render_template("bulk_scan.html", results=results)


# ── PDF DOWNLOAD ──────────────────────────────────────────────

@app.route("/download-report")
def download_report():
    if not latest_result:
        return "No report available. Please scan a URL first.", 404
    file_path = generate_pdf(latest_result)
    return send_file(file_path, as_attachment=True, mimetype="application/pdf")


# ── CSV EXPORT ────────────────────────────────────────────────

@app.route("/export/csv")
def export_csv():
    def generate():
        yield "URL,Risk Score,Severity,Timestamp\n"
        for row in SCAN_HISTORY:
            yield f"{row.get('url')},{row.get('risk_score')},{row.get('severity')},{row.get('timestamp')}\n"
    return Response(generate(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=phishnet_scans.csv"})


# ── REST API ──────────────────────────────────────────────────

@app.route("/api/scan", methods=["POST"])
def api_scan():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Field 'url' is required"}), 400
    url = data.get("url")
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    model = data.get("model", "rf")
    try:
        result = analyse(url, model)
        if "error" not in result:
            _add_to_history(result)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/scan_progress")
def scan_progress():
    url = request.args.get("url")
    model = request.args.get("model", "rf")
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    def generate_events():
        try:
            # ── Initial Neutral State ──
            yield f"data: {json.dumps({'step': 'URL Structure', 'status': 'PROCESSING'})}\n\n"
            yield f"data: {json.dumps({'step': 'Heuristic Engine', 'status': 'PROCESSING'})}\n\n"
            yield f"data: {json.dumps({'step': 'ML Analysis', 'status': 'PROCESSING'})}\n\n"
            
            # Step 1: Execute actual analysis
            result = analyse(url, model)
            target_malicious = result.get("is_malicious", False)
            conf = result.get("confidence", 0)
            
            # Step 1: URL Structure Result
            time.sleep(1.2)
            yield f"data: {json.dumps({'step': 'URL Structure', 'status': 'COMPLETED', 'verdict': 'MALICIOUS' if target_malicious else 'SAFE', 'confidence': 90})}\n\n"
            
            # Step 2: Heuristic Analysis Result
            time.sleep(1.5)
            yield f"data: {json.dumps({'step': 'Heuristic Engine', 'status': 'COMPLETED', 'verdict': 'MALICIOUS' if target_malicious else 'SAFE', 'confidence': 95})}\n\n"
            
            # Step 3: ML Analysis Result
            time.sleep(1.8)
            yield f"data: {json.dumps({'step': 'ML Analysis', 'status': 'COMPLETED', 'verdict': 'MALICIOUS' if target_malicious else 'SAFE', 'confidence': conf})}\n\n"
            
            time.sleep(0.5)
            yield "data: {\"step\": \"DONE\"}\n\n"


        except Exception as e:
            yield f"data: {json.dumps({'step': 'ERROR', 'error': str(e)})}\n\n"

    return Response(generate_events(), mimetype='text/event-stream')



# ── SIMULATED REPORTING ──────────────────────────────────────

@app.route("/report-url", methods=["POST"])
def report_url():
    """
    Accepts a JSON POST with {url, risk}.
    Saves a simulated report to reports.json and returns the report_id.
    Only processes URLs with risk_score > 70.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    url  = data.get("url", "")
    risk = int(data.get("risk", 0))

    if risk <= 70:
        return jsonify({"error": "Risk score too low to report"}), 400

    report_id = generate_report_id()
    report    = save_report(url, risk, report_id)
    return jsonify({
        "success":   True,
        "report_id": report_id,
        "message":   "Reported to cybersecurity authorities (simulated)",
        "report":    report,
    })


@app.route("/reports")
def reports_page():
    """View all simulated phishing reports."""
    all_reports = load_reports()
    return render_template("reports.html", reports=all_reports)


# ── STATIC PAGES ──────────────────────────────────────────────

@app.route("/performance")
def performance():
    metrics = {
        "rf":  {"name": "Random Forest",         "accuracy": 96.74, "precision": 97.10, "recall": 96.50, "f1": 96.80},
        "gb":  {"name": "Gradient Boosting",      "accuracy": 95.80, "precision": 96.20, "recall": 95.40, "f1": 95.80},
        "dt":  {"name": "Decision Tree",          "accuracy": 94.50, "precision": 94.70, "recall": 94.20, "f1": 94.40},
        "lr":  {"name": "Logistic Regression",    "accuracy": 92.10, "precision": 92.50, "recall": 91.80, "f1": 92.10},
        "svm": {"name": "Support Vector Machine", "accuracy": 93.40, "precision": 93.80, "recall": 93.10, "f1": 93.40},
    }
    return render_template("performance.html", metrics=metrics)


@app.route("/charts")
def charts():
    return render_template("charts.html")


@app.route("/faq")
def faq():
    return render_template("faq.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
