"""Expert Router — classifies prompts and routes to the appropriate expert."""

import os
import json
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer


class ExpertRouter:
    """Routes user prompts to the appropriate expert (code/math/general)."""

    def __init__(
        self,
        model_dir: str = "router/model",
        embedding_model_name: str = None,
        confidence_threshold: float = 0.5,
    ):
        self.model_dir = model_dir
        self.confidence_threshold = confidence_threshold

        # Load config
        config_path = os.path.join(model_dir, "router_config.json")
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Load embedding model
        model_name = embedding_model_name or self.config["embedding_model"]
        print(f"Loading router embedding model: {model_name}")
        self.encoder = SentenceTransformer(model_name)

        # Load classifier and label encoder
        self.classifier = joblib.load(os.path.join(model_dir, "router_classifier.joblib"))
        self.label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))

        print(f"Router loaded. Experts: {list(self.label_encoder.classes_)}")

    def route(self, prompt: str) -> dict:
        """Route a prompt to the best expert. Returns expert, confidence, and probabilities."""
        embedding = self.encoder.encode([prompt])
        probabilities = self.classifier.predict_proba(embedding)[0]

        predicted_idx = np.argmax(probabilities)
        predicted_label = self.label_encoder.inverse_transform([predicted_idx])[0]
        confidence = probabilities[predicted_idx]

        # Fall back to general if confidence is below threshold
        threshold_applied = False
        if confidence < self.confidence_threshold:
            predicted_label = "general"
            threshold_applied = True

        prob_map = {
            label: float(prob)
            for label, prob in zip(self.label_encoder.classes_, probabilities)
        }

        return {
            "expert": predicted_label,
            "confidence": float(confidence),
            "probabilities": prob_map,
            "threshold_applied": threshold_applied,
        }

    def route_batch(self, prompts: list) -> list:
        """Route multiple prompts at once (batched for efficiency)."""
        embeddings = self.encoder.encode(prompts)
        probabilities = self.classifier.predict_proba(embeddings)

        results = []
        for probs in probabilities:
            predicted_idx = np.argmax(probs)
            predicted_label = self.label_encoder.inverse_transform([predicted_idx])[0]
            confidence = probs[predicted_idx]

            threshold_applied = False
            if confidence < self.confidence_threshold:
                predicted_label = "general"
                threshold_applied = True

            prob_map = {
                label: float(prob)
                for label, prob in zip(self.label_encoder.classes_, probs)
            }

            results.append({
                "expert": predicted_label,
                "confidence": float(confidence),
                "probabilities": prob_map,
                "threshold_applied": threshold_applied,
            })

        return results


if __name__ == "__main__":
    router = ExpertRouter()

    print("\nROUTER DEMO — Type a prompt to see routing (quit to exit)")
    while True:
        prompt = input("\nYou: ").strip()
        if prompt.lower() in ("quit", "exit", "q"):
            break
        if not prompt:
            continue

        result = router.route(prompt)
        print(f"  Expert: {result['expert'].upper()} | Confidence: {result['confidence']:.1%}")
        for expert, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 20)
            print(f"    {expert:8s}: {prob:.3f} {bar}")
