# PhishNet v2.0 — Production-Ready Malicious URL Detector

PhishNet v2.0 is an industry-grade URL phishing detection system featuring a modular services architecture, multi-model support, and a 5-layer explainability engine. It combines high-accuracy Machine Learning with robust heuristic rules to provide transparent and actionable threat intelligence.

## 🚀 Key Features
- **5-Layer Detection**: Whitelist (L1), ML Models (L2), URL Structure (L3), Keywords/TLDs (L4), and Obfuscation (L5).
- **Unified Risk Scoring**: 0–100 score weighted between ML signal (65%) and heuristic rules (35%).
- **REST API**: Structured JSON endpoint (`POST /api/scan`) for programmatic integration.
- **Explainability Panel**: Visual breakdown of why a URL was flagged.
- **In-Memory History**: Tracks recent scan activity and site-wide statistics.
- **Premium UI**: Responsive glassmorphism dashboard with severity-based theme coding.

## 🏗️ Architecture
- `app.py`: Flask web interface and REST API routes.
- `services/detector.py`: Core detection engine (pure Python).
- `src/features.py`: URL feature extraction (25 unique features).
- `model/`: Serialized ML models (Random Forest, Gradient Boosting).
- `templates/`: Jinja2 templates for the dashboard, scanner, and history.

## ⚙️ Quick Start
1. `pip install -r requirements.txt`
2. `python train.py` (if models are not present)
3. `python app.py`
4. Visit `http://127.0.0.1:5000` (Login: `admin` / `admin`)
