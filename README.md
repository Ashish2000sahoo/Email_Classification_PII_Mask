# 📧 Email Classification & PII Masking API

This project implements a robust email classification system for customer support teams, incorporating PII masking to safeguard sensitive user data. The model classifies emails into categories such as Billing Issues, Technical Support, and more, and is deployed via a Streamlit-powered API.

## 🚀 Project Overview

The system supports the following capabilities:
- **PII Detection & Masking**: Automatically identifies and masks personal information (Full Name, Email, Phone, etc.) using regex and NLP without LLMs.
- **Email Classification**: Categorizes emails into support-related topics using a Naive Bayes classifier.
- **API Access**: Exposes functionality via a Streamlit interface, complying with strict JSON output format for integration and evaluation.

## 🛠️ Tech Stack

- Python, Pandas, Scikit-learn, NLTK
- Streamlit for UI and API
- Regex for PII detection
- Joblib for model serialization

## 📂 Project Structure

```plaintext
├── app.py               # Model training and email processing logic
├── api.py               # Streamlit API UI
├── models.py            # Classification pipeline definition
├── utils.py             # PII masking and text preprocessing utilities
├── requirements.txt     # Python dependencies
├── output/              # Contains trained model and label encoder
