# Extraction of linguistic features

import re
import spacy
import numpy as np
from numpy.linalg import norm
from collections import Counter
from wordfreq import zipf_frequency
from sentence_transformers import SentenceTransformer

nlp = spacy.load('en_core_web_sm')
transformer = SentenceTransformer('all-mpnet-base-v2')

def _split_sentences(text: str):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

# ========== FEATURE EXTRACTION ========== #

# Features 1-6: Basic text stats
def _text_stats(transcript: dict):
    text = transcript.get('text', '')
    words = re.findall(r'\b\w+\b', text)
    total_tokens = len(words)
    unique_tokens = len(set(words))
    type_token_ratio = unique_tokens / total_tokens if total_tokens else 0.0

    utterances = [len(re.findall(r'\b\w+\b', s.get('text', ''))) for s in transcript.get('segments', [])]
    if utterances:
        mean_words_per_utterance = float(np.mean(utterances))
        max_utterance_length = max(utterances)
    else:
        mean_words_per_utterance = 0.0
        max_utterance_length = 0

    sentences = _split_sentences(text)
    sentence_count = len(sentences)

    return total_tokens, unique_tokens, type_token_ratio, mean_words_per_utterance, max_utterance_length, sentence_count

# Features 7-9: Lexical richness
def _lexical_richness(transcript: dict):
    vague_words = {
        'thing', 'things', 'stuff', 'place', 'places', 'something', 'anything', 'everything', 'someone', 'anyone', 'everyone', 'that', 'this', 'these', 'those'
    }

    doc = nlp(transcript.get('text', ''))
    total_words = 0
    content_words = 0
    function_words = 0
    rare_words = 0

    for token in doc:
        if token.is_alpha:
            w = token.lemma_.lower()
            total_words += 1

            if token.pos_ in {"NOUN", "PROPN", "VERB", "ADJ", "ADV"} and w not in vague_words:
                content_words += 1
            elif token.pos_ in {'DET', 'PRON', 'ADP', 'CCONJ', 'SCONJ', 'AUX', 'PART'}:
                function_words += 1

            freq = zipf_frequency(w, 'en')
            if freq < 3.0:
                rare_words += 1

    content_words_ratio = content_words / total_words if total_words else 0
    function_words_ratio = function_words / total_words if total_words else 0
    rare_words_ratio = rare_words / total_words if total_words else 0
    return content_words_ratio, function_words_ratio, rare_words_ratio

# Features 10-13: Repetition & disfluency
def _repetition_disfluency(transcript: dict):
    text = transcript.get('text', '').lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    filler_count = transcript.get('filler_count', 0) + sum([text.count(f) for f in [
        'well', 'actually', 'basically', 'so', 'like', 'you know'
    ]])

    self_correction_count = sum([text.count(f) for f in [
        'i mean', 'sorry', 'i meant', 'wait', 'hold on', 'hang on', 'rephrase', 'start again', 'rather'
    ]])

    tokens = re.findall(r'\b\w+\b', transcript.get('text', ''))
    tokens = [t.lower() for t in tokens if t.strip()]
    total_tokens = len(tokens)

    repetition_score = 0.0
    bigram_repetition_ratio = 0

    for n in range(1, 6):  # looks for ngrams up to 5-grams
        if total_tokens < n:
            continue

        # build n-grams
        ngrams = [' '.join(tokens[i:i + n]) for i in range(total_tokens - n + 1)]
        counts = Counter(ngrams)

        # only look at ones that appear more than once
        repeated = {ng: c for ng, c in counts.items() if c > 1}

        if n == 2:
            raw_repeat = sum(c - 1 for c in repeated.values())
            total_ngrams = len(ngrams)

            bigram_repetition_ratio = raw_repeat / total_ngrams if total_ngrams > 0 else 0.0

        # length-weighted “punishment” for repetitions
        length_weighted = sum((c - 1) * n for c in repeated.values())

        repetition_score += length_weighted

    # normalize repetition score
    if total_tokens > 0:
        repetition_score /= total_tokens
    else:
        repetition_score = 0

    return filler_count, repetition_score, bigram_repetition_ratio, self_correction_count

# Features 14-15: Semantic coherence
def _semantic_coherence(transcript: dict):
    def clean(s: str) -> str:
        s = s.lower()
        s = re.sub(r'[^a-z0-9\s]', ' ', s)
        s = re.sub(r'\s+', ' ', s)
        s = s.strip()
        return s

    text = transcript.get('text', '')
    raw_sentences = _split_sentences(text)
    sentences = [clean(s) for s in raw_sentences if s.strip()]
    if len(sentences) < 2:
        return 0.0, 0.0

    embeddings = transformer.encode(sentences)
    similarities = []

    def cosine_similarity(v1, v2):
        dot_product = np.dot(v1, v2)
        magnitude_v1 = norm(v1)
        magnitude_v2 = norm(v2)
        if magnitude_v1 == 0 or magnitude_v2 == 0:
            return 0.0
        return dot_product / (magnitude_v1 * magnitude_v2)

    for i in range(len(embeddings) - 1):
        similarities.append(cosine_similarity(embeddings[i], embeddings[i + 1]))

    semantic_coherence_mean = float(np.mean(similarities))
    semantic_coherence_variance = float(np.var(similarities))

    return semantic_coherence_mean, semantic_coherence_variance

# ========== COMBINE EVERYTHING ========== #

def extract(transcript: dict):
    total_tokens, unique_tokens, type_token_ratio, mean_words_per_utterance, max_utterance_length, sentence_count = _text_stats(transcript)
    content_words_ratio, function_words_ratio, rare_words_ratio = _lexical_richness(transcript)
    filler_count, repetition_score, bigram_repetition_ratio, self_correction_count = _repetition_disfluency(transcript)
    semantic_coherence_mean, semantic_coherence_variance = _semantic_coherence(transcript)

    LINGUISTIC_FEATURES = np.array([
        # Basic text stats (6)
        total_tokens, unique_tokens, type_token_ratio, mean_words_per_utterance, max_utterance_length, sentence_count,

        # Lexical richness (3)
        content_words_ratio, function_words_ratio, rare_words_ratio,

        # Repetition & disfluency (4)
        filler_count, repetition_score, bigram_repetition_ratio, self_correction_count,

        # Semantic coherence (2)
        semantic_coherence_mean, semantic_coherence_variance
    ])

    return LINGUISTIC_FEATURES
