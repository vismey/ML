"""
features.py
Handcrafted statistical and linguistic feature extractor for AI vs Human text classification.
Extracts:
  1. Lexical Diversity (Type-Token Ratio)
  2. Average Sentence Length (Mean words per sentence)
  3. Sentence Length Variance / Standard Deviation ("Burstiness")
  4. Average Word Length
  5. Punctuation Frequencies (Commas, Semicolons, Exclamation/Question marks)
  6. Stopword Ratio
  7. Readability Indices (Flesch Reading Ease, Flesch-Kincaid Grade)
  8. Part-of-Speech (POS) Ratios (Nouns, Verbs, Adjectives, Adverbs)
  9. Formal Discourse & Transition Marker Frequency
"""

import re
import numpy as np
import pandas as pd

# NLTK imports with robust fallback handling
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk import pos_tag
    STOPWORDS = set(stopwords.words('english'))
except Exception:
    STOPWORDS = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with',
        'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
        'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if',
        'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him',
        'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than',
        'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two',
        'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us'
    }

# textstat import with standard fallback formulas
try:
    import textstat
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False

# Classic AI transitional / formal discourse markers often overused by LLMs
AI_DISCOURSE_MARKERS = [
    r'\bfurthermore\b', r'\bmoreover\b', r'\bconsequently\b', r'\bin addition\b',
    r'\bit is important to\b', r'\bit is worth noting\b', r'\bpivotal\b',
    r'\bcrucial\b', r'\bcomprehensive\b', r'\bparadigm\b', r'\bunderscores\b',
    r'\bdelve\b', r'\bseamlessly\b', r'\bfosters\b', r'\bharnessing\b',
    r'\btestament\b', r'\bin conclusion\b', r'\bultimately\b'
]


def simple_sentence_tokenize(text: str):
    """Splits text into sentences using regex boundary matching."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def simple_word_tokenize(text: str):
    """Extracts alphanumeric words from text."""
    return re.findall(r'\b[a-zA-Z0-9_\'-]+\b', text.lower())


def count_syllables_word(word: str) -> int:
    """Estimates syllable count using vowel clusters."""
    w = word.lower().strip()
    if len(w) <= 3:
        return 1
    w = re.sub(r'(?:[^laeiouy]|ed|es|e)$', '', w)
    w = re.sub(r'^y', '', w)
    matches = re.findall(r'[aeiouy]{1,2}', w)
    return max(1, len(matches))


def compute_flesch_reading_ease(text: str, words, sentences) -> float:
    """Computes Flesch Reading Ease score (0 to 100+)."""
    if HAS_TEXTSTAT:
        try:
            return float(textstat.flesch_reading_ease(text))
        except Exception:
            pass
    total_words = len(words)
    total_sentences = len(sentences)
    if total_words == 0 or total_sentences == 0:
        return 50.0
    total_syllables = sum(count_syllables_word(w) for w in words)
    # Standard formula: 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
    score = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
    return round(score, 2)


def compute_flesch_kincaid_grade(text: str, words, sentences) -> float:
    """Computes Flesch-Kincaid Grade Level."""
    if HAS_TEXTSTAT:
        try:
            return float(textstat.flesch_kincaid_grade(text))
        except Exception:
            pass
    total_words = len(words)
    total_sentences = len(sentences)
    if total_words == 0 or total_sentences == 0:
        return 8.0
    total_syllables = sum(count_syllables_word(w) for w in words)
    # Standard formula: 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
    score = 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59
    return max(0.0, round(score, 2))


def extract_features_dict(text: str) -> dict:
    """
    Extracts a dictionary of all 14 handcrafted statistical and linguistic features
    from an input text string.
    """
    text = text.strip() if text else ""
    if not text:
        return {
            "lexical_diversity": 0.0,
            "avg_sentence_length": 0.0,
            "sentence_length_std": 0.0,
            "avg_word_length": 0.0,
            "comma_freq": 0.0,
            "semicolon_freq": 0.0,
            "exclamation_freq": 0.0,
            "question_freq": 0.0,
            "stopword_ratio": 0.0,
            "flesch_reading_ease": 50.0,
            "flesch_kincaid_grade": 8.0,
            "noun_ratio": 0.0,
            "verb_ratio": 0.0,
            "adjective_ratio": 0.0,
            "adverb_ratio": 0.0,
            "ai_discourse_markers": 0.0
        }

    # Sentences and words
    try:
        sentences = sent_tokenize(text)
    except Exception:
        sentences = simple_sentence_tokenize(text)
    if not sentences:
        sentences = [text]

    try:
        raw_words = word_tokenize(text)
        words = [w.lower() for w in raw_words if re.match(r'^[a-zA-Z0-9_\'-]+$', w)]
    except Exception:
        words = simple_word_tokenize(text)

    num_words = max(1, len(words))
    num_sentences = max(1, len(sentences))

    # 1. Lexical Diversity (Type-Token Ratio: Unique Words / Total Words)
    unique_words = len(set(words))
    lexical_diversity = unique_words / num_words

    # 2 & 3. Sentence Length Statistics (Mean & Variance/Std Dev - Burstiness)
    sentence_word_counts = []
    for s in sentences:
        swords = simple_word_tokenize(s)
        sentence_word_counts.append(len(swords))

    avg_sentence_len = float(np.mean(sentence_word_counts))
    sentence_len_std = float(np.std(sentence_word_counts)) if len(sentence_word_counts) > 1 else 0.0

    # 4. Average Word Length (character count)
    avg_word_len = sum(len(w) for w in words) / num_words

    # 5. Punctuation Frequencies per sentence
    comma_count = text.count(',')
    semicolon_count = text.count(';') + text.count(':')
    exclamation_count = text.count('!')
    question_count = text.count('?')

    comma_freq = comma_count / num_sentences
    semicolon_freq = semicolon_count / num_sentences
    exclamation_freq = exclamation_count / num_sentences
    question_freq = question_count / num_sentences

    # 6. Stopword Ratio
    stopword_count = sum(1 for w in words if w in STOPWORDS)
    stopword_ratio = stopword_count / num_words

    # 7. Readability Metrics
    flesch_ease = compute_flesch_reading_ease(text, words, sentences)
    flesch_grade = compute_flesch_kincaid_grade(text, words, sentences)

    # 8. Part of Speech (POS) Tag Distributions
    noun_count = 0
    verb_count = 0
    adj_count = 0
    adv_count = 0

    try:
        # pos_tag from NLTK
        tags = pos_tag(words)
        for _, tag in tags:
            if tag.startswith('NN'):
                noun_count += 1
            elif tag.startswith('VB'):
                verb_count += 1
            elif tag.startswith('JJ'):
                adj_count += 1
            elif tag.startswith('RB'):
                adv_count += 1
    except Exception:
        # Fallback heuristic if POS tagger is unavailable
        noun_count = int(0.25 * num_words)
        verb_count = int(0.18 * num_words)
        adj_count = int(0.10 * num_words)
        adv_count = int(0.06 * num_words)

    noun_ratio = noun_count / num_words
    verb_ratio = verb_count / num_words
    adjective_ratio = adj_count / num_words
    adverb_ratio = adv_count / num_words

    # 9. AI Discourse Marker Frequency (per 100 words)
    marker_hits = 0
    lower_text = text.lower()
    for pattern in AI_DISCOURSE_MARKERS:
        marker_hits += len(re.findall(pattern, lower_text))
    ai_discourse_markers = (marker_hits / num_words) * 100.0

    return {
        "lexical_diversity": round(float(lexical_diversity), 4),
        "avg_sentence_length": round(float(avg_sentence_len), 3),
        "sentence_length_std": round(float(sentence_len_std), 3),
        "avg_word_length": round(float(avg_word_len), 3),
        "comma_freq": round(float(comma_freq), 3),
        "semicolon_freq": round(float(semicolon_freq), 3),
        "exclamation_freq": round(float(exclamation_freq), 3),
        "question_freq": round(float(question_freq), 3),
        "stopword_ratio": round(float(stopword_ratio), 4),
        "flesch_reading_ease": round(float(flesch_ease), 2),
        "flesch_kincaid_grade": round(float(flesch_grade), 2),
        "noun_ratio": round(float(noun_ratio), 4),
        "verb_ratio": round(float(verb_ratio), 4),
        "adjective_ratio": round(float(adjective_ratio), 4),
        "adverb_ratio": round(float(adverb_ratio), 4),
        "ai_discourse_markers": round(float(ai_discourse_markers), 3)
    }


def extract_features_dataframe(texts) -> pd.DataFrame:
    """Extracts features for an iterable of texts into a pandas DataFrame."""
    feature_rows = [extract_features_dict(t) for t in texts]
    return pd.DataFrame(feature_rows)


FEATURE_NAMES = [
    "lexical_diversity",
    "avg_sentence_length",
    "sentence_length_std",
    "avg_word_length",
    "comma_freq",
    "semicolon_freq",
    "exclamation_freq",
    "question_freq",
    "stopword_ratio",
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "noun_ratio",
    "verb_ratio",
    "adjective_ratio",
    "adverb_ratio",
    "ai_discourse_markers"
]

FEATURE_LABELS = {
    "lexical_diversity": "Lexical Diversity (Type-Token Ratio)",
    "avg_sentence_length": "Avg Sentence Length (words)",
    "sentence_length_std": "Sentence Length Std Dev (Burstiness)",
    "avg_word_length": "Avg Word Length (chars)",
    "comma_freq": "Comma Frequency per Sentence",
    "semicolon_freq": "Semicolon/Colon Frequency",
    "exclamation_freq": "Exclamation Mark Frequency",
    "question_freq": "Question Mark Frequency",
    "stopword_ratio": "Stopword Ratio",
    "flesch_reading_ease": "Flesch Reading Ease (0-100)",
    "flesch_kincaid_grade": "Flesch-Kincaid Grade Level",
    "noun_ratio": "Noun Ratio (POS)",
    "verb_ratio": "Verb Ratio (POS)",
    "adjective_ratio": "Adjective Ratio (POS)",
    "adverb_ratio": "Adverb Ratio (POS)",
    "ai_discourse_markers": "AI Formal Discourse Markers (per 100 words)"
}
