"""
Streamlit Web Interface for the SMS Spam Filter Mini Project.

Run locally:
    pip install streamlit pandas scikit-learn
    streamlit run streamlit_app.py
"""

import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from sms_spam_filter import build_pipeline, load_dataset, predict_message


st.set_page_config(page_title="SMS Spam Filter", page_icon="📩", layout="centered")
st.title("SMS Spam Filter")
st.caption(
    "Classify SMS messages as Spam or Ham using TF-IDF with Multinomial Naive Bayes "
    "and Logistic Regression."
)


@st.cache_resource
def train_models(data_path: str):
    df = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_message"],
        df["target"],
        test_size=0.25,
        random_state=42,
        stratify=df["target"],
    )

    models = {
        "Multinomial Naive Bayes": build_pipeline("nb"),
        "Logistic Regression": build_pipeline("logreg"),
    }
    metrics = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        metrics[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "cm": confusion_matrix(y_test, preds),
        }
    return len(df), models, metrics


data_file = st.sidebar.text_input("Dataset path", value="SMSSpamCollection")
model_choice = st.sidebar.selectbox(
    "Classifier",
    ["Multinomial Naive Bayes", "Logistic Regression"],
)

dataset_size, models, metrics = train_models(data_file)
st.sidebar.info(f"Loaded {dataset_size} messages.")

selected_pipeline = models[model_choice]
selected_metrics = metrics[model_choice]

st.subheader("Test a Message")
user_sms = st.text_area(
    "Enter an SMS message to classify:",
    placeholder="e.g., Congratulations! You won a free $500 voucher. Reply CLAIM now.",
    height=110,
)

if st.button("Classify Message", type="primary"):
    if not user_sms.strip():
        st.warning("Please enter a message first.")
    else:
        label, confidence = predict_message(selected_pipeline, user_sms)
        if label == "SPAM":
            st.error(f"Prediction: **SPAM** (confidence: {confidence:.1%})")
        else:
            st.success(f"Prediction: **HAM (Legitimate)** (confidence: {confidence:.1%})")

st.subheader(f"Evaluation Metrics ({model_choice})")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{selected_metrics['accuracy']:.2%}")
col2.metric("Precision", f"{selected_metrics['precision']:.2%}")
col3.metric("Recall", f"{selected_metrics['recall']:.2%}")
col4.metric("F1-Score", f"{selected_metrics['f1']:.2%}")

st.write("Confusion Matrix (rows = actual `[Ham, Spam]`, columns = predicted `[Ham, Spam]`):")
st.dataframe(selected_metrics["cm"])