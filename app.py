import os
import subprocess
import pandas as pd
import json
import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils import mask_pii, preprocess_text
from models import get_model

nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))

# Load dataset
df = pd.read_csv("combined_emails_with_natural_pii.csv")

# Mask PII
masked_data = [mask_pii(text) for text in df["email"]]
df["masked_email"] = [d["masked_text"] for d in masked_data]
df["masked_entities"] = [d["entities"] for d in masked_data]

# Preprocess text
df["clean_text"] = df["masked_email"].apply(preprocess_text)
df = df.dropna(subset=["clean_text", "type"])

# Encode labels
le = LabelEncoder()
df["label"] = le.fit_transform(df["type"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, random_state=42
)

# Train model
pipeline = get_model()
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Export predictions
# Ensure the output directory exists
output_dir = "output/json_outputs"
os.makedirs(output_dir, exist_ok=True)

# Save each prediction as a separate file
for i, row in df.iloc[X_test.index].iterrows():
    pred_label = le.inverse_transform([pipeline.predict([row["clean_text"]])[0]])[0]
    output = {
        "input_email_body": row["email"],
        "list_of_masked_entities": row["masked_entities"],
        "masked_email": row["masked_email"],
        "category_of_the_email": pred_label
    }

    with open(os.path.join(output_dir, f"{i}.json"), "w") as f:
        json.dump(output, f, indent=2)

# Save model & encoder for API use
import joblib
joblib.dump(pipeline, "output/email_classifier.joblib")
joblib.dump(le, "output/label_encoder.joblib")


if __name__ == "__main__":
    print("🔁 Running app.py logic...")

    # Your existing logic here (e.g. model training, data preprocessing, etc.)
    # ...

    print("🚀 Launching Streamlit app...")
    subprocess.run([
    "D:/FULL STACK DATA SCIENCE AND AI/AKAIKE ASSIGNMENT/venv/python.exe",
    "-m", "streamlit", "run", "api.py"
    ])