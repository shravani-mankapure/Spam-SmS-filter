"""
SMS Spam Filter Mini Project (TF-IDF + Naive Bayes & Logistic Regression)

Dataset:
    UCI SMS Spam Collection (5,574 labeled SMS messages: 'ham' or 'spam')
    Download: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
    Place the extracted 'SMSSpamCollection' file in the same folder as this script.
    If the file is not found locally, the script runs on a small built-in sample
    so you can verify the pipeline immediately.

Requirements:
    pip install pandas scikit-learn
"""

from pathlib import Path
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


SAMPLE_MESSAGES = [
    ("ham", "Hey, are we still meeting for lunch at 1 pm?"),
    ("ham", "I'll call you back as soon as I get home."),
    ("ham", "Don't forget to bring the lab report tomorrow morning."),
    ("ham", "Can you send me the notes from today's machine learning lecture?"),
    ("ham", "Happy birthday! Hope you have a great day ahead."),
    ("ham", "Running 10 minutes late, stuck in traffic."),
    ("ham", "Let me know once you reach the station safely."),
    ("ham", "Thanks for helping me debug the assignment earlier."),
    ("spam", "URGENT! You have won a $1,000 cash prize. Claim now by calling 09061701461."),
    ("spam", "FREE entry to win a brand new smartphone! Text WIN to 80085 now."),
    ("spam", "Congratulations! Your mobile number has been selected for a £500 gift voucher."),
    ("spam", "Limited time offer: Get 80% off luxury watches. Click http://bit.ly/deal-now"),
    ("spam", "You have 1 unread voicemail. Call 08712345678 immediately to listen."),
    ("spam", "Exclusive loan approval! Borrow up to $5,000 with zero paperwork today."),
    ("spam", "WINNER! Reply YES to claim your free holiday package to Dubai."),
    ("spam", "Your account has been suspended. Verify your details at http://verify-login.example"),
]


def clean_text(text: str) -> str:
    """Normalize whitespace and strip non-alphanumeric characters while keeping words."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\d+", " numtoken ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_dataset(filepath: str = "SMSSpamCollection") -> pd.DataFrame:
    """Load the UCI SMS Spam Collection TSV file, or fall back to sample data."""
    path = Path(filepath)
    if path.exists():
        df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"])
        print(f"Loaded {len(df)} messages from '{path}'.")
    else:
        df = pd.DataFrame(SAMPLE_MESSAGES, columns=["label", "message"])
        print(
            f"'{filepath}' not found. Running with {len(df)} built-in sample messages.\n"
            "Download the full dataset from: "
            "https://archive.ics.uci.edu/dataset/228/sms+spam+collection"
        )
    df["clean_message"] = df["message"].astype(str).apply(clean_text)
    df["target"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def build_pipeline(model_type: str = "nb") -> Pipeline:
    """Create a TF-IDF + classifier pipeline."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    if model_type == "nb":
        classifier = MultinomialNB(alpha=0.5)
    elif model_type == "logreg":
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
    else:
        raise ValueError("model_type must be 'nb' or 'logreg'")

    return Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier),
    ])


def evaluate_model(name: str, pipeline: Pipeline, X_test, y_test) -> None:
    """Print accuracy, confusion matrix, and classification report."""
    preds = pipeline.predict(X_test)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print("Confusion Matrix (rows=actual [ham, spam], cols=predicted):")
    print(confusion_matrix(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["ham", "spam"], zero_division=0))


def predict_message(pipeline: Pipeline, message: str) -> tuple[str, float]:
    """Predict whether a single SMS message is ham or spam with probability."""
    cleaned = clean_text(message)
    pred = pipeline.predict([cleaned])[0]
    prob = pipeline.predict_proba([cleaned])[0][pred]
    label = "SPAM" if pred == 1 else "HAM"
    return label, float(prob)


def main() -> None:
    df = load_dataset("SMSSpamCollection")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_message"],
        df["target"],
        test_size=0.25,
        random_state=42,
        stratify=df["target"],
    )

    nb_pipeline = build_pipeline("nb")
    nb_pipeline.fit(X_train, y_train)
    evaluate_model("Multinomial Naive Bayes", nb_pipeline, X_test, y_test)

    logreg_pipeline = build_pipeline("logreg")
    logreg_pipeline.fit(X_train, y_train)
    evaluate_model("Logistic Regression", logreg_pipeline, X_test, y_test)

    test_samples = [
        "Hey, are you free for coffee after class?",
        "CONGRATS! You won a free $500 voucher. Text CLAIM to 88088 now!",
    ]
    print("\n=== Sample Predictions (Naive Bayes) ===")
    for msg in test_samples:
        label, confidence = predict_message(nb_pipeline, msg)
        print(f"[{label} | {confidence:.2%}] {msg}")


if __name__ == "__main__":
    main()