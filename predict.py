#!/usr/bin/env python
"""
Fake News Detection CLI & Inference Tool
Usage:
    python predict.py "Your news text here..."
    python predict.py --file path/to/article.txt
"""
import sys
import os
import argparse
import json
import warnings

warnings.filterwarnings("ignore")

# Ensure UTF-8 stdout for Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.explainer import analyze_article


def main():
    parser = argparse.ArgumentParser(description="AI/ML Fake News & Content Reliability Detector")
    parser.add_argument("text", nargs="?", default=None, help="News article text or headline to analyze")
    parser.add_argument("--file", "-f", help="Path to text file containing article content")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()

    content = ""
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            content = f.read()
    elif args.text:
        content = args.text
    else:
        print("Reading text from standard input (Ctrl+Z then Enter to finish):")
        try:
            content = sys.stdin.read()
        except KeyboardInterrupt:
            return

    if not content.strip():
        print("Error: No text provided to analyze.")
        sys.exit(1)

    result = analyze_article(content)

    if args.json:
        # Exclude html markup in json summary
        clean_res = {k: v for k, v in result.items() if k != "highlighted_html"}
        print(json.dumps(clean_res, indent=2))
        return

    print("=" * 65)
    print("           NEWS DETECTIVE: AI RELIABILITY REPORT           ")
    print("=" * 65)
    print(f"VERDICT:            [{result['verdict'].upper()}] - {result['tier']}")
    print(f"CONFIDENCE SCORE:   {result['confidence_score']}%")
    print(f"RELIABLE PROB:      {result['prob_reliable']}%")
    print(f"MISLEADING PROB:    {result['prob_misleading']}%")
    print("-" * 65)
    print("LINGUISTIC & TEXTUAL PATTERNS:")
    lp = result["linguistic_patterns"]
    print(f"  • Word Count:          {lp['word_count']}")
    print(f"  • Sensationalism Index: {lp['sensationalism_score']} / 100")
    print(f"  • Caps/Shouting Ratio: {round(lp['caps_ratio']*100, 1)}%")
    print(f"  • Exclamation Marks:   {lp['exclamation_count']}")
    print(f"  • Subjectivity Score:  {lp['subjectivity']} (0=Objective, 1=Subjective)")
    print(f"  • Reading Ease Score:  {lp['reading_ease']} / 100")
    if lp["sensational_trigger_matches"]:
        print(f"  • Clickbait Triggers:  {', '.join(lp['sensational_trigger_matches'])}")

    print("-" * 65)
    print("KEY INDICATORS (TOP INFLUENTIAL WORDS):")
    if result["top_reliable_indicators"]:
        print("  Positive (Reliability-correlated markers):")
        for item in result["top_reliable_indicators"][:5]:
            print(f"    + {item['word']:<18} (score: +{item['score']:.3f})")
    if result["top_misleading_indicators"]:
        print("  Negative (Misleading-correlated markers):")
        for item in result["top_misleading_indicators"][:5]:
            print(f"    - {item['word']:<18} (score: {item['score']:.3f})")

    print("-" * 65)
    print("EXPLAINABILITY & REASONING:")
    for r in result["reasoning"]:
        print(f"  • {r}")
    print("=" * 65)


if __name__ == "__main__":
    main()
