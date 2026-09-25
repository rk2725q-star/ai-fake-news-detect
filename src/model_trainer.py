import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import clean_text_for_nlp

DATA_DIR = os.path.join(BASE_DIR, "Datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_and_preprocess_data(sample_limit: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads True.csv and Fake.csv, cleans text, handles nulls, and formats labels:
    - 1 = Reliable / True
    - 0 = Misleading / Fake
    """
    fake_path = os.path.join(DATA_DIR, "Fake.csv")
    true_path = os.path.join(DATA_DIR, "True.csv")

    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        zip_path = os.path.join(DATA_DIR, "datasets.zip")
        if os.path.exists(zip_path):
            import zipfile
            print(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(DATA_DIR)
        else:
            raise FileNotFoundError(f"Missing Fake.csv or True.csv and datasets.zip in {DATA_DIR}")

    print("Loading Fake.csv and True.csv...")
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)

    df_fake["label"] = 0
    df_fake["label_name"] = "Misleading"

    df_true["label"] = 1
    df_true["label_name"] = "Reliable"

    if sample_limit:
        df_fake = df_fake.sample(n=min(sample_limit // 2, len(df_fake)), random_state=42)
        df_true = df_true.sample(n=min(sample_limit // 2, len(df_true)), random_state=42)

    # Combine title and text to provide rich contextual signals
    df_fake["full_text"] = df_fake["title"].fillna("") + " " + df_fake["text"].fillna("")
    df_true["full_text"] = df_true["title"].fillna("") + " " + df_true["text"].fillna("")

    df = pd.concat([df_fake, df_true], ignore_index=True)
    df = df.dropna(subset=["full_text"])
    df = df[df["full_text"].str.strip().str.len() > 10]

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Total dataset size after cleaning: {len(df)} samples ({len(df_fake)} Fake, {len(df_true)} True)")

    return df


def train_and_evaluate(sample_limit: int = None) -> Dict[str, Any]:
    os.makedirs(MODELS_DIR, exist_ok=True)
    start_time = time.time()

    df = load_and_preprocess_data(sample_limit=sample_limit)

    print("Cleaning text with NLP normalizer (removing datelines, URLs, formatting)...")
    clean_texts = [clean_text_for_nlp(t) for t in df["full_text"]]
    labels = df["label"].values

    print("Performing stratified train/test split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        clean_texts, labels, test_size=0.20, random_state=42, stratify=labels
    )

    print("Fitting TF-IDF Vectorizer strictly on train set...")
    vectorizer = TfidfVectorizer(
        max_features=25000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3,
        max_df=0.85,
        stop_words="english"
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"TF-IDF feature shape: {X_train_vec.shape}")

    # Model definitions to compare
    candidate_models = {
        "Logistic Regression": LogisticRegression(C=2.0, max_iter=1000, random_state=42),
        "Passive Aggressive": CalibratedClassifierCV(
            estimator=PassiveAggressiveClassifier(max_iter=1000, random_state=42),
            cv=3
        ),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Random Forest": RandomForestClassifier(n_estimators=60, max_depth=25, random_state=42, n_jobs=-1)
    }

    results = {}
    best_name = None
    best_f1 = -1.0
    best_model = None

    for name, clf in candidate_models.items():
        print(f"--- Training {name} ---")
        t0 = time.time()
        clf.fit(X_train_vec, y_train)
        train_time = round(time.time() - t0, 2)

        preds = clf.predict(X_test_vec)
        probs = clf.predict_proba(X_test_vec)[:, 1] if hasattr(clf, "predict_proba") else None

        acc = round(accuracy_score(y_test, preds), 4)
        prec = round(precision_score(y_test, preds), 4)
        rec = round(recall_score(y_test, preds), 4)
        f1 = round(f1_score(y_test, preds), 4)
        cm = confusion_matrix(y_test, preds).tolist()
        report = classification_report(y_test, preds, target_names=["Misleading", "Reliable"], output_dict=True)

        print(f"{name} Results -> Accuracy: {acc*100:.2f}%, F1: {f1*100:.2f}%, Train Time: {train_time}s")

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "train_time_sec": train_time,
            "confusion_matrix": cm,
            "classification_report": report
        }

        # Keep Logistic Regression as primary explainer model because of direct interpretable feature coefficients
        if name == "Logistic Regression" or (best_model is None and f1 > best_f1):
            best_name = name
            best_f1 = f1
            best_model = clf

    # Extract top global indicators from Logistic Regression
    lr_model = candidate_models["Logistic Regression"]
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = lr_model.coef_[0]

    # Top positive coefficients indicate Reliable (class 1)
    top_reliable_idx = np.argsort(coefs)[-30:][::-1]
    top_reliable_indicators = [
        {"word": str(feature_names[i]), "weight": round(float(coefs[i]), 4)}
        for i in top_reliable_idx
    ]

    # Top negative coefficients indicate Misleading (class 0)
    top_fake_idx = np.argsort(coefs)[:30]
    top_fake_indicators = [
        {"word": str(feature_names[i]), "weight": round(float(coefs[i]), 4)}
        for i in top_fake_idx
    ]

    total_duration = round(time.time() - start_time, 2)

    bundle = {
        "primary_model": lr_model,
        "all_models": {k: candidate_models[k] for k in ["Logistic Regression", "Passive Aggressive", "Multinomial Naive Bayes"]},
        "vectorizer": vectorizer,
        "feature_names": feature_names.tolist(),
        "lr_coefficients": coefs.tolist(),
        "top_reliable_indicators": top_reliable_indicators,
        "top_fake_indicators": top_fake_indicators,
        "model_performance": results,
        "training_metadata": {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_count": len(feature_names),
            "training_duration_sec": total_duration,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }

    bundle_path = os.path.join(MODELS_DIR, "fake_news_detector.joblib")
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")

    print(f"Saving model bundle to {bundle_path}...")
    joblib.dump(bundle, bundle_path, compress=3)

    metrics_summary = {
        "model_performance": results,
        "training_metadata": bundle["training_metadata"],
        "top_reliable_indicators": top_reliable_indicators[:15],
        "top_fake_indicators": top_fake_indicators[:15]
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print("Model training, evaluation and persistence complete!")
    return metrics_summary


if __name__ == "__main__":
    train_and_evaluate()
