import streamlit as st
import joblib
import json
from utils import mask_pii, preprocess_text

# Configure dark mode
st.set_page_config(page_title="Email Classifier", layout="centered", initial_sidebar_state="collapsed")

# Load model and label encoder
model = joblib.load("output/email_classifier.joblib")
label_encoder = joblib.load("output/label_encoder.joblib")

# UI
st.markdown("<h1 style='text-align: center;'>📧 Email Classifier with PII Masking</h1>", unsafe_allow_html=True)
st.markdown("Paste a raw email below, and click **Classify Email** to mask personal data and categorize the email.")

# Input box
email_input = st.text_area("✉️ Enter Raw Email", height=250)

# Button
if st.button("🔍 Classify Email"):
    if not email_input.strip():
        st.warning("⚠️ Please paste an email first.")
    else:
        try:
            # Step 1: Mask
            result = mask_pii(email_input)
            masked_email = result["masked_text"]
            masked_entities = result["entities"]

            # Step 2: Preprocess
            processed_text = preprocess_text(masked_email)

            # Step 3: Predict
            prediction = model.predict([processed_text])[0]
            predicted_label = label_encoder.inverse_transform([prediction])[0]

            # Final output
            output = {
                "input_email_body": email_input,
                "list_of_masked_entities": masked_entities,
                "masked_email": masked_email,
                "category_of_the_email": predicted_label
            }

            # Display results
            st.success("✅ Classification Successful!")
            st.json(output)

            # JSON download button
            st.download_button(
                label="⬇️ Download Result as JSON",
                data=json.dumps(output, indent=2),
                file_name="classified_email.json",
                mime="application/json"
            )

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")