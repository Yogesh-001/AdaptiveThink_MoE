"""Train the prompt router (MiniLM embeddings + LogisticRegression)."""

import json
import os
import sys
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


def load_training_data(data_path: str) -> tuple:
    """Load router training data (text + labels)."""
    with open(data_path, "r") as f:
        data = json.load(f)
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    return texts, labels


def train_router(
    data_path: str = "router/router_training_data.json",
    output_dir: str = "router/model",
    embedding_model: str = "all-MiniLM-L6-v2",
):
    """Train the router classifier on labeled prompts."""
    print("=" * 60)
    print("TRAINING ROUTER")
    print("=" * 60)

    # Load data
    print(f"\nLoading training data from: {data_path}")
    texts, labels = load_training_data(data_path)
    print(f"  Total samples: {len(texts)}")
    print(f"  Distribution: { {l: labels.count(l) for l in set(labels)} }")

    # Encode prompts into embeddings
    print(f"\nLoading embedding model: {embedding_model}")
    encoder = SentenceTransformer(embedding_model)

    print("Encoding prompts...")
    embeddings = encoder.encode(texts, show_progress_bar=True)
    print(f"  Embedding shape: {embeddings.shape}")

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")

    # Train classifier
    print("\nTraining LogisticRegression...")
    classifier = LogisticRegression(C=10, max_iter=1000, random_state=42)
    classifier.fit(X_train, y_train)

    # Evaluate
    print(f"\n{'=' * 60}")
    print("EVALUATION")
    print(f"{'=' * 60}")

    test_accuracy = classifier.score(X_test, y_test)
    print(f"\nTest Accuracy: {test_accuracy*100:.1f}%")

    cv_scores = cross_val_score(classifier, embeddings, y, cv=5, scoring="accuracy")
    print(f"Cross-validation: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

    y_pred = classifier.predict(X_test)
    print(f"\n{classification_report(y_test, y_pred, target_names=label_encoder.classes_)}")

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(classifier, os.path.join(output_dir, "router_classifier.joblib"))
    joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.joblib"))

    config = {
        "embedding_model": embedding_model,
        "embedding_dim": 384,
        "n_classes": len(label_encoder.classes_),
        "classes": list(label_encoder.classes_),
        "test_accuracy": float(test_accuracy),
        "cv_accuracy_mean": float(cv_scores.mean()),
    }
    with open(os.path.join(output_dir, "router_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"ROUTER TRAINING COMPLETE — Saved to: {output_dir}")
    print(f"{'=' * 60}")

    # Demo
    print(f"\n{'─' * 60}")
    print("DEMO: Sample routing")
    print(f"{'─' * 60}")

    demo_prompts = [
        "Write a function to sort a dictionary by values",
        "What is 15% of 300?",
        "Explain how photosynthesis works",
        "Implement a binary tree traversal",
        "If I have 5 apples and give away 2, how many are left?",
        "What are the benefits of exercise?",
    ]

    demo_embeddings = encoder.encode(demo_prompts)
    demo_predictions = classifier.predict(demo_embeddings)
    demo_probabilities = classifier.predict_proba(demo_embeddings)

    for prompt, pred_idx, probs in zip(demo_prompts, demo_predictions, demo_probabilities):
        label = label_encoder.inverse_transform([pred_idx])[0]
        print(f"\n  \"{prompt}\"")
        print(f"  → {label} ({probs.max()*100:.1f}%)")


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "router/router_training_data.json"
    train_router(data_path=data_path)
