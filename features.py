"""
features.py
Linguistic and statistical feature extraction for AI vs. Human text classification.

Converts raw text into 16 numerical measurements across 8 categories:
  1. Lexical Diversity (Type-Token Ratio)
  2. Sentence Length & Burstiness (Mean & Standard Deviation)
  3. Average Word Length
  4. Punctuation Frequencies (Commas, Semicolons, Exclamation & Question Marks)
  5. Stopword Ratio
  6. Readability Indices (Flesch Reading Ease & Grade Level)
  7. Part-of-Speech Distributions (Nouns, Verbs, Adjectives, Adverbs)
  8. AI Formal Discourse Marker Frequency
"""

import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag
import textstat

# ---------------------------------------------------------------------------
# Setup Linguistic Resources
# ---------------------------------------------------------------------------
try:
    STOPWORDS = set(stopwords.words('english'))
except Exception:
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('punkt', quiet=True)
    STOPWORDS = set(stopwords.words('english'))

# Transition words and phrases frequently overused by LLMs
AI_DISCOURSE_MARKERS = [
    r'\bfurthermore\b', r'\bmoreover\b', r'\bconsequently\b', r'\bin addition\b',
    r'\bit is important to\b', r'\bit is worth noting\b', r'\bpivotal\b',
    r'\bcrucial\b', r'\bcomprehensive\b', r'\bparadigm\b', r'\bunderscores\b',
    r'\bdelve\b', r'\bseamlessly\b', r'\bfosters\b', r'\bharnessing\b',
    r'\btestament\b', r'\bin conclusion\b', r'\bultimately\b'
]


# ---------------------------------------------------------------------------
# Helper Tokenization Functions
# ---------------------------------------------------------------------------
def split_into_sentences(text: str) -> list:
    """Splits text into sentences based on punctuation (. ! ?)."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def split_into_words(text: str) -> list:
    """Extracts clean lowercase words (alphanumeric)."""
    return re.findall(r'\b[a-zA-Z0-9_\'-]+\b', text.lower())


# ---------------------------------------------------------------------------
# Main Feature Extraction Function
# ---------------------------------------------------------------------------
def extract_features_dict(text: str) -> dict:
    """
    Extracts 16 handcrafted linguistic and statistical features from a text string.
    Returns a dictionary mapping feature names to numerical values.
    """
    text = text.strip() if text else ""

    # Default values for empty input
    if not text:
        return {name: 0.0 for name in FEATURE_NAMES}

    sentences = split_into_sentences(text) or [text]
    words = split_into_words(text)
    num_words = max(1, len(words))
    num_sentences = max(1, len(sentences))

    # 1. Lexical Diversity (Unique words / Total words)
    # Measures vocabulary richness. Low diversity indicates repetitive wording.
    lexical_diversity = len(set(words)) / num_words

    # 2. Sentence Length Mean & Burstiness (Standard Deviation)
    # Humans vary sentence lengths widely (burstiness), whereas AI is more uniform.
    words_per_sentence = [len(split_into_words(s)) for s in sentences]
    avg_sentence_len = float(np.mean(words_per_sentence))
    sentence_len_std = float(np.std(words_per_sentence)) if len(sentences) > 1 else 0.0

    # 3. Average Word Length (Characters per word)
    avg_word_len = sum(len(w) for w in words) / num_words

    # 4. Punctuation Frequencies (Count per sentence)
    # Humans use conversational punctuation (! and ?); AI favors structured punctuation (, and ;).
    comma_freq = text.count(',') / num_sentences
    semicolon_freq = (text.count(';') + text.count(':')) / num_sentences
    exclamation_freq = text.count('!') / num_sentences
    question_freq = text.count('?') / num_sentences

    # 5. Stopword Ratio
    # Proportion of filler words ('the', 'is', 'and') relative to total words.
    stopword_count = sum(1 for w in words if w in STOPWORDS)
    stopword_ratio = stopword_count / num_words

    # 6. Readability Scores
    # Assesses reading complexity based on sentence length and syllables per word.
    try:
        flesch_ease = float(textstat.flesch_reading_ease(text))
        flesch_grade = float(textstat.flesch_kincaid_grade(text))
    except Exception:
        flesch_ease, flesch_grade = 50.0, 8.0

    # 7. Part-of-Speech (POS) Ratios
    # Measures syntactic distribution (Nouns, Verbs, Adjectives, Adverbs).
    noun_count, verb_count, adj_count, adv_count = 0, 0, 0, 0
    try:
        tagged_words = pos_tag(words)
        for _, tag in tagged_words:
            if tag.startswith('NN'):
                noun_count += 1
            elif tag.startswith('VB'):
                verb_count += 1
            elif tag.startswith('JJ'):
                adj_count += 1
            elif tag.startswith('RB'):
                adv_count += 1
    except Exception:
        pass

    noun_ratio = noun_count / num_words
    verb_ratio = verb_count / num_words
    adjective_ratio = adj_count / num_words
    adverb_ratio = adv_count / num_words

    # 8. AI Formal Discourse Marker Frequency (per 100 words)
    # Frequency of stereotypical LLM transition words (furthermore, delve, paradigm, etc.).
    lower_text = text.lower()
    marker_hits = sum(len(re.findall(pattern, lower_text)) for pattern in AI_DISCOURSE_MARKERS)
    ai_discourse_markers = (marker_hits / num_words) * 100.0

    # Return structured feature vector
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
    """Extracts features for a collection of texts into a pandas DataFrame."""
    try:
        from joblib import Parallel, delayed
        feature_rows = Parallel(n_jobs=-1, batch_size=50)(
            delayed(extract_features_dict)(t) for t in texts
        )
    except Exception:
        feature_rows = [extract_features_dict(t) for t in texts]
    return pd.DataFrame(feature_rows)


# ---------------------------------------------------------------------------
# Feature Metadata for Display & Model Inputs
# ---------------------------------------------------------------------------
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
