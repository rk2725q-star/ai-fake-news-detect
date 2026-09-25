import re
import string
from typing import Dict, Any, List

# List of common sensationalist/clickbait trigger phrases
SENSATIONAL_TRIGGERS = [
    r"\bshocking\b",
    r"\bbombshell\b",
    r"\bexposed\b",
    r"\bunbelievable\b",
    r"\bconspiracy\b",
    r"\bwake up\b",
    r"\bsecret\b",
    r"\bhoax\b",
    r"\bhidden truth\b",
    r"\bthey don'?t want you to know\b",
    r"\byou won'?t believe\b",
    r"\bmainstream media\b",
    r"\bdeep state\b",
    r"\bcover-?up\b",
    r"\bviral\b",
    r"\bmust-?see\b",
    r"\bcatastrophic\b",
    r"\bdisaster\b",
    r"\btreason\b",
    r"\bproof\b",
    r"\bpanic\b",
    r"\blow-?life\b",
    r"\bclueless\b"
]

DATELINE_REGEX = re.compile(
    r"^(?:[A-Z\s]{2,25}(?:\([A-Za-z\s]+\)|[A-Z\s]+)?\s*[-–—:]\s*(?:\(Reuters\)\s*[-–—:]?\s*)?|\(Reuters\)\s*[-–—:]\s*)",
    re.IGNORECASE
)

def clean_text_for_nlp(text: str, remove_datelines: bool = True) -> str:
    """
    Cleans raw text for machine learning featurization:
    - Removes HTML tags
    - Strips publisher datelines/wire attribution if specified
    - Removes URLs
    - Normalizes punctuation and numbers
    - Normalizes whitespace
    """
    if not isinstance(text, str):
        return ""

    t = text.strip()
    
    # Remove HTML tags
    t = re.sub(r"<[^>]+>", " ", t)

    # Remove news agency datelines at the beginning (e.g. "WASHINGTON (Reuters) - ")
    if remove_datelines:
        t = DATELINE_REGEX.sub("", t)
        t = re.sub(r"^\s*\(reuters\)\s*[-–—:]\s*", "", t, flags=re.IGNORECASE)

    # Remove URLs
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)

    # Remove square brackets and their contents
    t = re.sub(r"\[.*?\]", " ", t)

    # Remove non-word characters except spaces
    t = re.sub(r"[^\w\s]", " ", t)

    # Remove isolated numbers or digits within words
    t = re.sub(r"\b\d+\b", " ", t)

    # Lowercase and clean multiple whitespace
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()

    return t


def extract_linguistic_patterns(raw_text: str) -> Dict[str, Any]:
    """
    Extracts deep linguistic and textual indicators:
    - Character and word counts
    - Uppercase shouting ratio (indicator of sensationalism/clickbait)
    - Punctuation extremes (multiple ! and ?)
    - Clickbait & sensationalist phrase counts
    - Polarity and subjectivity estimates
    - Overall sensationalism index (0 - 100)
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return {
            "char_count": 0,
            "word_count": 0,
            "sentence_count": 0,
            "caps_ratio": 0.0,
            "all_caps_words": 0,
            "exclamation_count": 0,
            "question_count": 0,
            "sensational_trigger_matches": [],
            "sensational_trigger_count": 0,
            "polarity": 0.0,
            "subjectivity": 0.0,
            "sensationalism_score": 0.0,
            "reading_ease": 100.0,
        }

    char_count = len(raw_text)
    words = re.findall(r"\b\w+\b", raw_text)
    word_count = len(words)
    
    # Sentences
    sentences = [s.strip() for s in re.split(r"[.!?]+", raw_text) if s.strip()]
    sentence_count = max(len(sentences), 1)

    # Uppercase analysis (excluding 1-letter words like 'I' or 'A')
    multi_char_words = [w for w in words if len(w) > 1]
    all_caps_words = sum(1 for w in multi_char_words if w.isupper())
    letters_only = [c for c in raw_text if c.isalpha()]
    total_letters = len(letters_only)
    uppercase_letters = sum(1 for c in letters_only if c.isupper())
    caps_ratio = (uppercase_letters / total_letters) if total_letters > 0 else 0.0

    # Punctuation analysis
    exclamation_count = raw_text.count("!")
    question_count = raw_text.count("?")

    # Sensational triggers
    trigger_matches = []
    lower_text = raw_text.lower()
    for pattern in SENSATIONAL_TRIGGERS:
        found = re.findall(pattern, lower_text)
        if found:
            # normalize matched word
            match_name = pattern.replace(r"\b", "").replace(r"\?", "").replace(r"-?", "-")
            trigger_matches.append(match_name)
    
    # Sentiment & Subjectivity
    try:
        from textblob import TextBlob
        blob = TextBlob(raw_text[:2000])  # sample first 2000 chars for speed
        polarity = round(blob.sentiment.polarity, 3)
        subjectivity = round(blob.sentiment.subjectivity, 3)
    except Exception:
        polarity = 0.0
        subjectivity = 0.5

    # Reading ease estimation (simplified Flesch Reading Ease)
    # 206.835 - 1.015 * (total words / total sentences) - 84.6 * (total syllables / total words)
    syllable_count = 0
    for w in words:
        w_lower = w.lower()
        syl = len(re.findall(r"[aeiouy]+", w_lower))
        syllable_count += max(syl, 1)

    if word_count > 0 and sentence_count > 0:
        asl = word_count / sentence_count
        asw = syllable_count / word_count
        flesch = 206.835 - (1.015 * asl) - (84.6 * asw)
        reading_ease = max(0.0, min(100.0, round(flesch, 1)))
    else:
        reading_ease = 70.0

    # Composite Sensationalism Score (0 to 100)
    # Factors:
    # 1. High caps ratio (>0.08)
    # 2. Exclamation marks per 100 words
    # 3. Number of clickbait triggers
    # 4. Extreme subjectivity (>0.6)
    sensational_points = 0.0
    if caps_ratio > 0.08:
        sensational_points += min(30.0, (caps_ratio - 0.08) * 300)
    
    exclamation_rate = (exclamation_count / max(word_count, 1)) * 100
    sensational_points += min(25.0, exclamation_rate * 10)

    sensational_points += min(25.0, len(trigger_matches) * 8.0)

    if subjectivity > 0.55:
        sensational_points += (subjectivity - 0.55) * 40.0

    sensationalism_score = round(max(0.0, min(100.0, sensational_points)), 1)

    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "caps_ratio": round(caps_ratio, 4),
        "all_caps_words": all_caps_words,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "sensational_trigger_matches": trigger_matches,
        "sensational_trigger_count": len(trigger_matches),
        "polarity": polarity,
        "subjectivity": subjectivity,
        "sensationalism_score": sensationalism_score,
        "reading_ease": reading_ease,
    }
