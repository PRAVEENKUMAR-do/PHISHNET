"""
services/reporter.py — Simulated Cybersecurity Reporting for PhishNet
"""
import json, random, os
from datetime import datetime


def generate_report_id() -> str:
    """Generate a unique simulated report ID."""
    return "PHISH-" + str(random.randint(1000, 9999))


def save_report(url: str, risk: int, report_id: str) -> dict:
    """
    Save a phishing report locally to reports.json.
    Returns the saved report dict.
    """
    report = {
        "report_id": report_id,
        "url":       url,
        "risk":      risk,
        "timestamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "status":    "Reported (Simulated)",
    }

    reports_file = "reports.json"

    # Append as newline-delimited JSON (one record per line)
    with open(reports_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(report) + "\n")

    return report


def load_reports() -> list:
    """Load all saved reports from reports.json."""
    if not os.path.exists("reports.json"):
        return []
    reports = []
    with open("reports.json", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    reports.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return reports
