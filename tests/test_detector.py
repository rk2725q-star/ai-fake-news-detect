import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import clean_text_for_nlp, extract_linguistic_patterns
from src.explainer import analyze_article, load_model_bundle
from src.live_news import fetch_live_headlines
from sample_articles import SAMPLE_ARTICLES


class TestFakeNewsDetectionSystem(unittest.TestCase):

    def test_text_cleaning(self):
        raw = "WASHINGTON (Reuters) - The Senate approved the bill <p>online at https://example.com</p>."
        cleaned = clean_text_for_nlp(raw)
        self.assertNotIn("reuters", cleaned)
        self.assertNotIn("https", cleaned)
        self.assertNotIn("<p>", cleaned)
        self.assertIn("senate", cleaned)

    def test_linguistic_patterns_sensational(self):
        text = "SHOCKING BOMBSHELL! WAKE UP PEOPLE! You won't believe what happened!!!"
        stats = extract_linguistic_patterns(text)
        self.assertGreater(stats["sensationalism_score"], 40.0)
        self.assertGreater(stats["caps_ratio"], 0.3)
        self.assertGreaterEqual(stats["exclamation_count"], 3)
        self.assertGreaterEqual(len(stats["sensational_trigger_matches"]), 2)

    def test_linguistic_patterns_neutral(self):
        text = "The committee held its regular session on Tuesday morning to discuss annual budget allocations."
        stats = extract_linguistic_patterns(text)
        self.assertLess(stats["sensationalism_score"], 25.0)
        self.assertEqual(stats["exclamation_count"], 0)
        self.assertLess(stats["subjectivity"], 0.4)

    def test_model_bundle_loading(self):
        bundle = load_model_bundle()
        self.assertIn("primary_model", bundle)
        self.assertIn("vectorizer", bundle)
        self.assertIn("feature_names", bundle)
        self.assertIn("lr_coefficients", bundle)
        self.assertGreater(len(bundle["feature_names"]), 1000)

    def test_analyze_empty_input(self):
        res = analyze_article("")
        self.assertIn("error", res)

    def test_analyze_misleading_sample(self):
        text = SAMPLE_ARTICLES["[Misleading] BOMBSHELL: Secret Deep-State Microchip Program Exposed By Whistleblower!"]
        res = analyze_article(text)
        self.assertEqual(res["verdict"], "Misleading")
        self.assertGreater(res["confidence_score"], 80.0)
        self.assertIn("highlighted_html", res)
        self.assertGreater(len(res["top_misleading_indicators"]), 0)
        self.assertGreater(len(res["reasoning"]), 0)

    def test_analyze_reliable_sample(self):
        text = SAMPLE_ARTICLES["[Reliable] Global Renewable Energy Investment Hits Record High"]
        res = analyze_article(text)
        self.assertEqual(res["verdict"], "Reliable")
        self.assertGreater(res["prob_reliable"], 50.0)
        self.assertIn("highlighted_html", res)
        self.assertGreater(len(res["reasoning"]), 0)

    def test_probability_consistency(self):
        text = "Federal regulators completed their scheduled quarterly audit of regional banking reserves."
        res = analyze_article(text)
        total_prob = res["prob_reliable"] + res["prob_misleading"]
        self.assertAlmostEqual(total_prob, 100.0, places=1)

    def test_fetch_live_headlines(self):
        headlines = fetch_live_headlines(limit=3)
        self.assertGreater(len(headlines), 0)
        first = headlines[0]
        self.assertIn("title", first)
        self.assertIn("verdict", first)
        self.assertIn("confidence_score", first)
        self.assertIn(first["verdict"], ["Reliable", "Misleading"])


if __name__ == "__main__":
    unittest.main()
