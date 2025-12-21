# Extraction of linguistic features

import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np
import spacy
import textstat
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordfreq import zipf_frequency

nlp = spacy.load('en_core_web_sm')
transformer = SentenceTransformer('all-mpnet-base-v2')

def _split_sentences(text: str):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def _load_concreteness_lexicon():
    path = Path(__file__).with_name('concreteness.csv')
    concreteness = {}

    with path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        cols = reader.fieldnames
        word_col = cols[0]
        conc_col = cols[1]

        for row in reader:
            w = row[word_col].strip().lower()
            s = row[conc_col].strip()

            if not s:
                continue

            try:
                concreteness[w] = float(s)
            except ValueError:
                continue

    return concreteness

CONCRETENESS_MAP = _load_concreteness_lexicon()

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

    filler_count = transcript.get('fillers', 0) + sum([text.count(f) for f in [
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

# Features 16-18: Syntactic complexity
def _syntactic_complexity(transcript: dict):
    doc = nlp(transcript.get('text', ''))

    # mean dependency distance
    dependency_distances = []
    for sent in doc.sents:
        for token in sent:
            if token.head != token:
                distance = abs(token.i - token.head.i)
                dependency_distances.append(distance)
    mean_dependency_distance = float(np.mean(dependency_distances)) if dependency_distances else 0.0

    # clause density
    clause_count = 0
    sentence_count = 0
    for sent in doc.sents:
        sentence_count += 1
        clause_count += 1  # Main clause
        for token in sent:
            if token.dep_ in ['ccomp', 'xcomp', 'advcl', 'acl', 'relcl']:
                clause_count += 1
    clause_density = clause_count / sentence_count if sentence_count > 0 else 0.0

    # mean parse tree height
    def get_tree_height(token):
        if not list(token.children):
            return 1
        return 1 + max(get_tree_height(child) for child in token.children)

    tree_heights = [get_tree_height(sent.root) for sent in doc.sents]
    mean_parse_tree_height = float(np.mean(tree_heights)) if tree_heights else 0.0

    return mean_dependency_distance, clause_density, mean_parse_tree_height

# Features 19-21: Parts-of-speech ratios
def _pos_ratios(transcript: dict):
    doc = nlp(transcript.get('text', ''))

    total_words = 0
    pronoun_count = 0
    noun_count = 0
    verb_count = 0
    aux_verb_count = 0

    for token in doc:
        if token.is_alpha:
            total_words += 1
            if token.pos_ == 'PRON':
                pronoun_count += 1
            elif token.pos_ in ['NOUN', 'PROPN']:
                noun_count += 1
            elif token.pos_ == 'VERB':
                verb_count += 1
            elif token.pos_ == 'AUX':
                aux_verb_count += 1

    pronoun_ratio = pronoun_count / total_words if total_words else 0.0
    verb_to_noun_ratio = verb_count / noun_count if noun_count else 0.0
    auxiliary_verb_ratio = aux_verb_count / total_words if total_words else 0.0

    return pronoun_ratio, verb_to_noun_ratio, auxiliary_verb_ratio

# Features 22-24: Semantic content
def _semantic_content(transcript: dict):
    doc = nlp(transcript.get('text', ''))

    # idea density (count propositions)
    proposition_count = 0
    word_count = 0

    for token in doc:
        if token.is_alpha:
            word_count += 1
            if token.pos_ in ['VERB', 'ADJ', 'ADV', 'ADP', 'CCONJ', 'SCONJ']:
                proposition_count += 1

    idea_density = proposition_count / word_count if word_count else 0.0

    # mean concreteness score
    concreteness_scores = []
    for token in doc:
        if token.is_alpha and token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ']:
            lemma = token.lemma_.lower()
            score = CONCRETENESS_MAP.get(lemma)
            if score is not None:
                concreteness_scores.append(score)

    mean_concreteness = float(np.mean(concreteness_scores)) if concreteness_scores else 0.0

    # abstract word ratio
    def is_abstract(lemma: str, threshold: float = 2.0) -> bool:
        score = CONCRETENESS_MAP.get(lemma)
        if score is None:
            return False
        return score < threshold

    abstract_count = sum(
        1 for token in doc
        if token.is_alpha and is_abstract(token.lemma_.lower())
    )

    abstract_ratio = abstract_count / word_count if word_count > 0 else 0.0

    return idea_density, mean_concreteness, abstract_ratio

# Features 25-27: Vocabulary sophistication
def _vocabulary_sophistication(transcript: dict):
    text = transcript.get('text', '')

    # flesch-kincaid grade level
    fk_grade = textstat.flesch_kincaid_grade(text) if text else 0.0

    # mean syllables per word
    words = re.findall(r'\b\w+\b', text)
    syllable_counts = [textstat.syllable_count(word) for word in words]
    mean_syllables = float(np.mean(syllable_counts)) if syllable_counts else 0.0

    # long word ratio
    long_words = sum(1 for word in words if len(word) > 6)
    long_word_ratio = long_words / len(words) if words else 0.0

    return fk_grade, mean_syllables, long_word_ratio

# Features 28-29: Discourse coherence
def _discourse_coherence(transcript: dict):
    text = transcript.get('text', '')
    sentences = _split_sentences(text)

    if len(sentences) < 3:
        return 0.0, 0.0

    embeddings = transformer.encode(sentences)

    # global coherence drift
    # compare first sentence to all others
    first_embedding = embeddings[0]
    similarities_to_first = []

    for i in range(1, len(embeddings)):
        sim = np.dot(first_embedding, embeddings[i]) / (norm(first_embedding) * norm(embeddings[i]))
        similarities_to_first.append(sim)

    # decline in similarity over time
    if len(similarities_to_first) > 1:
        # use linear regression slope as drift measure
        x = np.arange(len(similarities_to_first))
        slope = np.polyfit(x, similarities_to_first, 1)[0]
        global_coherence_drift = float(-slope)
    else:
        global_coherence_drift = 0.0

    # topic recurrence score
    # tf-idf to find main topics and count recurrence
    if len(sentences) >= 2:
        vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            # how many sentences share the same topic
            feature_names = vectorizer.get_feature_names_out()

            # top topic per sentence
            topic_counts = Counter()
            for i in range(tfidf_matrix.shape[0]):
                row = tfidf_matrix[i].toarray()[0]
                if row.max() > 0:
                    top_topic_idx = row.argmax()
                    topic_counts[feature_names[top_topic_idx]] += 1

            # how often topics repeat
            total_occurrences = sum(topic_counts.values())
            recurrences = sum(count - 1 for count in topic_counts.values() if count > 1)
            topic_recurrence = recurrences / total_occurrences if total_occurrences else 0.0
        except:
            topic_recurrence = 0.0
    else:
        topic_recurrence = 0.0

    return global_coherence_drift, topic_recurrence

# ========== COMBINE EVERYTHING ========== #

def extract(transcript: dict, verbose=False):
    if verbose:
        print('[LING] Extracting linguistic features')

    total_tokens, unique_tokens, type_token_ratio, mean_words_per_utterance, max_utterance_length, sentence_count = _text_stats(transcript)
    content_words_ratio, function_words_ratio, rare_words_ratio = _lexical_richness(transcript)
    filler_count, repetition_score, bigram_repetition_ratio, self_correction_count = _repetition_disfluency(transcript)
    semantic_coherence_mean, semantic_coherence_variance = _semantic_coherence(transcript)
    mean_dependency_distance, clause_density, mean_parse_tree_height = _syntactic_complexity(transcript)
    pronoun_ratio, verb_to_noun_ratio, auxiliary_verb_ratio = _pos_ratios(transcript)
    idea_density, mean_concreteness, abstract_ratio = _semantic_content(transcript)
    fk_grade, mean_syllables, long_word_ratio = _vocabulary_sophistication(transcript)
    global_coherence_drift, topic_recurrence = _discourse_coherence(transcript)

    LINGUISTIC_FEATURES = np.array([
        # Basic text stats (6)
        total_tokens, unique_tokens, type_token_ratio, mean_words_per_utterance, max_utterance_length, sentence_count,

        # Lexical richness (3)
        content_words_ratio, function_words_ratio, rare_words_ratio,

        # Repetition & disfluency (4)
        filler_count, repetition_score, bigram_repetition_ratio, self_correction_count,

        # Semantic coherence (2)
        semantic_coherence_mean, semantic_coherence_variance,

        # Syntactic complexity (3)
        mean_dependency_distance, clause_density, mean_parse_tree_height,

        # Parts-of-speech ratios (3)
        pronoun_ratio, verb_to_noun_ratio, auxiliary_verb_ratio,

        # Semantic content (3)
        idea_density, mean_concreteness, abstract_ratio,

        # Vocabulary sophistication (3)
        fk_grade, mean_syllables, long_word_ratio,

        # Discourse coherence (2)
        global_coherence_drift, topic_recurrence
    ])

    if verbose:
        print('[LING] Done extracting')

    # replace all nans with zeros just for training purposes
    return np.nan_to_num(
        LINGUISTIC_FEATURES,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )
