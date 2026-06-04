"""
Router Training Script
========================

Trains a lightweight classifier that routes incoming prompts to the
correct expert (Code, Math, or General).

Architecture:
--------------
1. Sentence Embedding: all-MiniLM-L6-v2 (converts text → 384-dim vector)
2. Classifier: sklearn LogisticRegression (simple, fast, interpretable)

Why this architecture?
-----------------------

Option A (what we use): Embeddings + Classifier
- Fast: MiniLM encodes in ~5ms, LogReg predicts in <1ms
- Lightweight: No GPU needed for routing
- Interpretable: Can inspect decision boundaries, see confidence scores
- Easy to extend: Add new expert = add labeled examples + retrain classifier

Option B (alternative): Fine-tune a small classifier head on the LLM
- More complex, requires GPU for routing
- Overkill for 3 classes with distinct domains

Option C (alternative): LLM-based routing (ask GPT to classify)
- Expensive (full LLM inference just for routing)
- Adds latency before actual generation

We chose Option A because it's the best trade-off of simplicity,
speed, and accuracy for our use case.

Why all-MiniLM-L6-v2?
-----------------------
- Only 80MB (vs 400MB+ for larger models)
- Produces excellent semantic embeddings
- 384 dimensions (compact but expressive)
- Trained on 1B+ sentence pairs
- Fast inference even on CPU
- State-of-the-art for its size class
- Paper: "Sentence-BERT" (Reimers & Gurevych, 2019)

Why LogisticRegression (not neural network)?
---------------------------------------------
- 3 classes, 384 features, ~300 samples → LogReg is MORE than enough
- No overfitting risk (unlike a deep network on small data)
- Trains in milliseconds
- Provides probability scores (confidence) natively
- Scikit-learn implementation is battle-tested
- If needed later, can upgrade to a small MLP
"""

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
    """
    Load router training data (text + labels).

    Returns
    -------
    texts : list[str]
        The prompt texts
    labels : list[str]
        The expert labels ("code", "math", "general")
    """
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
    """
    Train the router classifier.

    Steps:
    1. Load labeled prompts
    2. Encode all prompts into embeddings using MiniLM
    3. Split into train/test
    4. Train LogisticRegression classifier
    5. Evaluate with cross-validation
    6. Save the model

    What gets saved:
    - router_classifier.joblib: The trained LogReg model
    - label_encoder.joblib: Maps between string labels and integers
    - The embedding model is NOT saved (loaded from sentence-transformers at inference)
    """

    print("=" * 60)
    print("TRAINING ROUTER")
    print("=" * 60)

    # 1. LOAD DATA
    print(f"\nLoading training data from: {data_path}")
    texts, labels = load_training_data(data_path)
    print(f"  Total samples: {len(texts)}")
    print(f"  Labels: {set(labels)}")
    print(f"  Distribution: { {l: labels.count(l) for l in set(labels)} }")

    # 2. ENCODE PROMPTS INTO EMBEDDINGS
    print(f"\nLoading embedding model: {embedding_model}")
    # SentenceTransformer loads the model and provides .encode() method
    # This converts variable-length text → fixed-size 384-dim vector
    encoder = SentenceTransformer(embedding_model)

    print("Encoding prompts into embeddings...")
    # encode() processes all texts and returns a numpy array
    # Shape: (n_samples, 384)
    embeddings = encoder.encode(texts, show_progress_bar=True)
    print(f"  Embedding shape: {embeddings.shape}")
    # embeddings[i] is a 384-dimensional vector representing texts[i]

    # 3. ENCODE LABELS
    # LabelEncoder: "code"→0, "math"→1, "general"→2 (or similar mapping)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    print(f"  Label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

    # 4. SPLIT DATA
    # 80% train, 20% test — stratified to keep class balance
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y,
        test_size=0.2,
        random_state=42,
        stratify=y,  # Ensures equal proportion of each class in both splits
    )
    print(f"\n  Train size: {len(X_train)}")
    print(f"  Test size: {len(X_test)}")

    # 5. TRAIN CLASSIFIER
    print("\nTraining LogisticRegression classifier...")
    # LogisticRegression parameters:
    # - C=10: Regularization strength (higher = less regularization)
    #   We use higher C because our data is clean and we want tight boundaries
    # - max_iter=1000: Maximum optimization iterations (ensure convergence)
    # - solver='lbfgs': Default solver, handles multinomial natively
    classifier = LogisticRegression(
        C=10,
        max_iter=1000,
        random_state=42,
    )
    classifier.fit(X_train, y_train)

    # 6. EVALUATE
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    # Test set accuracy
    test_accuracy = classifier.score(X_test, y_test)
    print(f"\nTest Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")

    # Cross-validation (more robust estimate)
    # 5-fold: split data into 5 parts, train on 4, test on 1, rotate
    cv_scores = cross_val_score(classifier, embeddings, y, cv=5, scoring="accuracy")
    print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Detailed classification report
    y_pred = classifier.predict(X_test)
    print(f"\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
    ))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"  (rows=true, cols=predicted)")
    print(f"  Labels: {list(label_encoder.classes_)}")
    print(f"  {cm}")

    # 7. SAVE MODEL
    os.makedirs(output_dir, exist_ok=True)

    classifier_path = os.path.join(output_dir, "router_classifier.joblib")
    encoder_path = os.path.join(output_dir, "label_encoder.joblib")

    joblib.dump(classifier, classifier_path)
    joblib.dump(label_encoder, encoder_path)

    # Also save the embedding model name for reference
    config = {
        "embedding_model": embedding_model,
        "embedding_dim": 384,
        "n_classes": len(label_encoder.classes_),
        "classes": list(label_encoder.classes_),
        "test_accuracy": float(test_accuracy),
        "cv_accuracy_mean": float(cv_scores.mean()),
    }
    config_path = os.path.join(output_dir, "router_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"ROUTER TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Saved:")
    print(f"  Classifier: {classifier_path}")
    print(f"  Label encoder: {encoder_path}")
    print(f"  Config: {config_path}")

    # 8. DEMO: Test with sample prompts
    print(f"\n{'─' * 60}")
    print("DEMO: Routing sample prompts")
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
        predicted_label = label_encoder.inverse_transform([pred_idx])[0]
        confidence = probs.max() * 100
        print(f"\n  \"{prompt}\"")
        print(f"  → {predicted_label} (confidence: {confidence:.1f}%)")
        # Show all class probabilities
        for cls, prob in zip(label_encoder.classes_, probs):
            bar = "█" * int(prob * 20)
            print(f"    {cls:8s}: {prob:.3f} {bar}")


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "router/router_training_data.json"
    train_router(data_path=data_path)
