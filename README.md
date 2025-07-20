# AI-Powered Auth-Log Analyzer

A cybersecurity tool that analyzes Linux `/var/log/auth.log` lines using machine learning:
- 🧠 Unsupervised anomaly detection (Isolation Forest)
- 🛡️ Streamlit UI for file upload and threat visualization

## Tech Stack
- FastAPI
- Scikit-learn
- Streamlit
- Hugging Face Transformers

## Getting Started
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
streamlit run frontend/streamlit_app.py
```
