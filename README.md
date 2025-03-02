
# IDENTIFICATION OF PERSONALLY IDENTIFIABLE INFORMATION (PII) IN DOCUMENTS AND DATA

**Problem Statement ID**: PS10  
**Project Category**: Data Privacy & Security

## 🔍 Overview
AI-powered system for detecting and redacting Personally Identifiable Information (PII) in documents. Supports Aadhaar, PAN, Driving License, and other sensitive documents with real-time alerts and secure masking capabilities.

## 🚀 Features
- Automatic PII detection using Gemini AI
- Document preview with redaction capabilities
- Secure user-specific document storage
- Audit logging and CSV exports
- Multi-format support (PDF/JPEG/PNG)

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- Tesseract OCR ([Installation guide](https://tesseract-ocr.github.io/tessdoc/Installation.html))

### Setup
1. Clone repository:
   ```bash
   git clone https://github.com/yourusername/pii-detection-system.git
   cd pii-detection-system

2. Install dependencies:
   pip install -r requirements.txt

3. Create .env file:
   touch .env


🔑 Environment Configuration
Add these to your .env file:
GEMINI_API_KEY=your_api_key_here
FLASK_SECRET_KEY=your_secret_key_here

Obtaining Keys
Gemini API Key:

Visit Google AI Studio

Create new API key

Copy key to .env

Flask Secret Key:
Generate using Python:
python -c "import secrets; print(secrets.token_hex(24))"

Copy output to FLASK_SECRET_KEY

🖥️ Running the Application
# Create necessary directories
mkdir -p uploads data

# Start application
python app.py

Access at: http://localhost:5000

📂 Project Structure
.
├── app.py                 # Main application
├── requirements.txt       # Dependencies
├── .env                   # Environment variables
├── uploads/               # User document storage
├── data/                  # Audit logs and CSV data
├── static/                # CSS/JS assets
└── templates/             # HTML templates

🤝 Contributing
1. Fork the repository
2. Create feature branch:
   git checkout -b feature/AmazingFeature
3. Commit changes:
   git commit -m 'Add some AmazingFeature'
4. Push to branch:
   git push origin feature/AmazingFeature
5. Open Pull Request

📄 License
Distributed under MIT License. See LICENSE for more information.