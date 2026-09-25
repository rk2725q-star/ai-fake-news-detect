import os
import sys
import html
import joblib
import numpy as np
from typing import Dict, Any, List, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import clean_text_for_nlp, extract_linguistic_patterns

_CACHED_BUNDLE = None


def load_model_bundle():
    global _CACHED_BUNDLE
    if _CACHED_BUNDLE is not None:
        return _CACHED_BUNDLE

    bundle_path = os.path.join(BASE_DIR, "models", "fake_news_detector.joblib")
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(
            f"Trained model not found at {bundle_path}. Please run src/model_trainer.py first."
        )

    _CACHED_BUNDLE = joblib.load(bundle_path)
    return _CACHED_BUNDLE


def analyze_article(text: str) -> Dict[str, Any]:
    """
    Comprehensive analysis pipeline:
    1. Text preprocessing & linguistic pattern analysis
    2. ML model prediction with calibrated probabilities
    3. Multi-model consensus
    4. Feature-level attribution (Explainable AI / Key Indicators)
    5. HTML token highlighting with contribution weights
    6. Linguistic flags & reasoning summary
    """
    if not text or not text.strip():
        return {
            "error": "Empty text provided. Please enter a news article or headline."
        }

    bundle = load_model_bundle()
    vectorizer = bundle["vectorizer"]
    primary_model = bundle["primary_model"]
    all_models = bundle.get("all_models", {})
    feature_names = np.array(bundle["feature_names"])
    coefs = np.array(bundle["lr_coefficients"])

    # 1. Linguistic Patterns
    linguistic_stats = extract_linguistic_patterns(text)

    # 2. Preprocess text for ML
    clean_text = clean_text_for_nlp(text)
    if not clean_text:
        return {
            "error": "Text contains no valid alphanumeric words after cleaning."
        }

    # 3. TF-IDF vectorization
    vec_input = vectorizer.transform([clean_text])
    
    # 4. Multi-model predictions
    model_predictions = {}
    for name, clf in all_models.items():
        try:
            pred = int(clf.predict(vec_input)[0])
            prob = clf.predict_proba(vec_input)[0] if hasattr(clf, "predict_proba") else None
            prob_reliable = float(prob[1]) if prob is not None else (1.0 if pred == 1 else 0.0)
            model_predictions[name] = {
                "prediction": "Reliable" if pred == 1 else "Misleading",
                "prob_reliable": round(prob_reliable, 4),
                "prob_misleading": round(1.0 - prob_reliable, 4)
            }
        except Exception:
            pass

    # Primary Model Output (Logistic Regression)
    primary_probs = primary_model.predict_proba(vec_input)[0]
    prob_misleading = float(primary_probs[0])
    prob_reliable = float(primary_probs[1])

    is_reliable = prob_reliable >= 0.50
    confidence = prob_reliable if is_reliable else prob_misleading
    predicted_label = "Reliable" if is_reliable else "Misleading"

    # Reliability Tier Classification
    if prob_reliable >= 0.85:
        tier = "Highly Reliable"
        verdict_color = "#10b981"  # Emerald green
        badge_variant = "success"
    elif prob_reliable >= 0.60:
        tier = "Likely Reliable"
        verdict_color = "#059669"
        badge_variant = "success"
    elif prob_reliable >= 0.40:
        tier = "Uncertain / Mixed Signals"
        verdict_color = "#f59e0b"  # Amber
        badge_variant = "warning"
    elif prob_reliable >= 0.15:
        tier = "Likely Misleading"
        verdict_color = "#f97316"  # Orange
        badge_variant = "danger"
    else:
        tier = "Highly Misleading / Deceptive"
        verdict_color = "#ef4444"  # Red
        badge_variant = "danger"

    # 5. Explainability & Key Indicators (Feature-level contribution)
    # Get active non-zero features in this input
    cx = vec_input.tocoo()
    word_attributions = []
    
    # Map feature index to word and calculate score = tfidf * coefficient
    for idx, val in zip(cx.col, cx.data):
        word = feature_names[idx]
        coef = coefs[idx]
        contrib = val * coef
        word_attributions.append({
            "word": word,
            "tfidf": round(float(val), 4),
            "coef": round(float(coef), 4),
            "score": round(float(contrib), 4),
            "direction": "Reliable" if contrib > 0 else "Misleading"
        })

    # Sort indicators by magnitude of contribution
    top_reliable_tokens = sorted(
        [w for w in word_attributions if w["score"] > 0],
        key=lambda x: x["score"],
        reverse=True
    )[:10]

    top_misleading_tokens = sorted(
        [w for w in word_attributions if w["score"] < 0],
        key=lambda x: x["score"]
    )[:10]

    # Map words to scores for text highlighting
    score_lookup = {w["word"]: w["score"] for w in word_attributions}

    # 6. Generate Highlighted HTML Text
    highlighted_html = _generate_highlighted_text(text, score_lookup)

    # 7. Reasoning & Risk Indicators
    reasoning_points = []
    if linguistic_stats["sensationalism_score"] > 35:
        reasoning_points.append(
            f"⚠️ High Sensationalism Score ({linguistic_stats['sensationalism_score']}/100): "
            f"The text exhibits heightened emotional style or clickbait characteristics."
        )
    if linguistic_stats["caps_ratio"] > 0.08:
        reasoning_points.append(
            f"📢 Excessive Capitalization: {round(linguistic_stats['caps_ratio']*100, 1)}% of characters are uppercase, "
            f"often associated with tabloid urgency or outrage bait."
        )
    if linguistic_stats["exclamation_count"] >= 3:
        reasoning_points.append(
            f"❗ Punctuation Outliers: Contains {linguistic_stats['exclamation_count']} exclamation marks."
        )
    if linguistic_stats["sensational_trigger_matches"]:
        triggers = ", ".join(linguistic_stats["sensational_trigger_matches"][:4])
        reasoning_points.append(
            f"🎯 Clickbait / Provocative Triggers Detected: Found phrases such as [{triggers}]."
        )
    if linguistic_stats["subjectivity"] > 0.60:
        reasoning_points.append(
            f"💭 High Subjectivity ({linguistic_stats['subjectivity']}): Text expresses personal opinions or emotionally charged rhetoric rather than neutral reporting."
        )
    elif linguistic_stats["subjectivity"] < 0.35 and is_reliable:
        reasoning_points.append(
            f"⚖️ Objective Tone: Low subjectivity score ({linguistic_stats['subjectivity']}) consistent with neutral journalistic reporting."
        )

    if is_reliable and top_reliable_tokens:
        sample_words = ", ".join([w["word"] for w in top_reliable_tokens[:4]])
        reasoning_points.append(
            f"✅ Authentic Vocabulary Correlates: Strong statistical alignment with credible reporting terminology ({sample_words})."
        )
    elif not is_reliable and top_misleading_tokens:
        sample_words = ", ".join([w["word"] for w in top_misleading_tokens[:4]])
        reasoning_points.append(
            f"❌ Deceptive / Misleading Markers: Significant alignment with known misleading vocabulary patterns ({sample_words})."
        )

    if not reasoning_points:
        reasoning_points.append("Standard neutral vocabulary detected across analyzed sentences.")

    return {
        "verdict": predicted_label,
        "confidence_score": round(confidence * 100, 2),
        "prob_reliable": round(prob_reliable * 100, 2),
        "prob_misleading": round(prob_misleading * 100, 2),
        "tier": tier,
        "verdict_color": verdict_color,
        "badge_variant": badge_variant,
        "model_consensus": model_predictions,
        "top_reliable_indicators": top_reliable_tokens,
        "top_misleading_indicators": top_misleading_tokens,
        "linguistic_patterns": linguistic_stats,
        "reasoning": reasoning_points,
        "highlighted_html": highlighted_html,
        "cleaned_text": clean_text
    }


def _generate_highlighted_text(raw_text: str, score_lookup: Dict[str, float]) -> str:
    """
    Renders safe HTML with tokens styled according to their impact:
    - Green background for words predicting Reliability
    - Red/Pink background for words predicting Misleading
    """
    tokens = raw_text.split()
    html_parts = []

    for token in tokens:
        clean_w = token.lower().strip(".,!?:;\"'()[]{}<>-")
        score = score_lookup.get(clean_w, 0.0)
        safe_token = html.escape(token)

        if score > 0.08:
            # Positive indicator (Reliable)
            intensity = min(1.0, score * 1.5)
            alpha = max(0.2, min(0.65, intensity))
            html_parts.append(
                f'<span style="background-color: rgba(34, 197, 94, {alpha:.2f}); color: inherit; padding: 2px 4px; border-radius: 4px; font-weight: 500;" title="Reliability indicator: +{score:.3f}">{safe_token}</span>'
            )
        elif score < -0.08:
            # Negative indicator (Misleading)
            intensity = min(1.0, abs(score) * 1.5)
            alpha = max(0.2, min(0.65, intensity))
            html_parts.append(
                f'<span style="background-color: rgba(239, 68, 68, {alpha:.2f}); color: inherit; padding: 2px 4px; border-radius: 4px; font-weight: 500;" title="Misleading marker: {score:.3f}">{safe_token}</span>'
            )
        else:
            html_parts.append(safe_token)

    return " ".join(html_parts)
