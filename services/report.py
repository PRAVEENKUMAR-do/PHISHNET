from fpdf import FPDF
from datetime import datetime
import os

def generate_pdf(data):
    pdf = FPDF()
    pdf.add_page()

    # ── Header ────────────────────────────────────────────────
    pdf.set_fill_color(10, 20, 50)
    pdf.rect(0, 0, 210, 30, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 30, "PhishNet Security Report", ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # ── URL ───────────────────────────────────────────────────
    pdf.set_font("Arial", "B", 12)
    pdf.cell(40, 8, "URL:", ln=False)
    pdf.set_font("Arial", "", 12)
    # Sanitise URL — strip non-latin chars that fpdf latin-1 can't encode
    url_str = data.get('url', 'N/A').encode('latin-1', errors='replace').decode('latin-1')
    pdf.cell(0, 8, url_str, ln=True)

    # ── Result ────────────────────────────────────────────────
    result_str = data.get('result', 'N/A')
    pdf.set_font("Arial", "B", 12)
    pdf.cell(40, 8, "Result:", ln=False)
    if result_str == "PHISHING":
        pdf.set_text_color(220, 50, 50)
    else:
        pdf.set_text_color(34, 139, 34)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, result_str, ln=True)
    pdf.set_text_color(0, 0, 0)

    # ── Risk Score ────────────────────────────────────────────
    pdf.set_font("Arial", "B", 12)
    pdf.cell(40, 8, "Risk Score:", ln=False)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"{data.get('risk', 0)}%", ln=True)
    pdf.ln(3)

    # ── Feature Analysis ──────────────────────────────────────
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Feature Analysis:", ln=True)
    pdf.set_font("Arial", "", 11)
    for key, value in data.get('features', {}).items():
        key_s   = str(key).encode('latin-1', errors='replace').decode('latin-1')
        value_s = str(value).encode('latin-1', errors='replace').decode('latin-1')
        pdf.cell(0, 7, f"  - {key_s}: {value_s}", ln=True)
    pdf.ln(3)

    # ── Explanation ───────────────────────────────────────────
    reasons = data.get('reasons', [])
    if reasons:
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, "Why Flagged as Phishing:", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.set_text_color(180, 0, 0)
        for r in reasons:
            r_s = str(r).encode('latin-1', errors='replace').decode('latin-1')
            pdf.cell(0, 7, f"  * {r_s}", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

    # ── Model Comparison ──────────────────────────────────────
    models = data.get('model_results', [])
    if models:
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, "Model Comparison:", ln=True)
        pdf.set_font("Arial", "", 11)
        for m in models:
            label = "PHISHING" if m.get('is_malicious') else "SAFE"
            line = f"  {m.get('name','?')}: {label} (confidence {m.get('confidence',0)}%)"
            pdf.cell(0, 7, line.encode('latin-1', errors='replace').decode('latin-1'), ln=True)
        pdf.ln(3)

    # ── Timestamp ─────────────────────────────────────────────
    pdf.set_font("Arial", "I", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 7, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

    # ── Save ──────────────────────────────────────────────────
    os.makedirs("static", exist_ok=True)
    file_path = os.path.join("static", "report.pdf")
    pdf.output(file_path)
    return file_path
