# Extraction of linguistic features

from __future__ import annotations

import csv
import math
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

from utils import cache
from utils.language import normalize_language

try:
    import jieba
    import jieba.posseg as jieba_posseg
except Exception:  # pragma: no cover - optional dependency
    jieba = None
    jieba_posseg = None

try:
    from pypinyin import lazy_pinyin
except Exception:  # pragma: no cover - optional dependency
    lazy_pinyin = None

CACHE_DIR = Path('cache/linguistics')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

EN_MODEL = 'en_core_web_sm'
ZH_MODEL = 'zh_core_web_sm'

SENTENCE_MODELS = {
    'en': 'all-mpnet-base-v2',
    'zh': 'paraphrase-multilingual-mpnet-base-v2',
}

CONTENT_POS = {'NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV'}
FUNCTION_POS = {'DET', 'PRON', 'ADP', 'CCONJ', 'SCONJ', 'AUX', 'PART'}

EN_VAGUE_WORDS = {
    'thing', 'things', 'stuff', 'place', 'places', 'something', 'anything',
    'everything', 'someone', 'anyone', 'everyone', 'that', 'this', 'these',
    'those'
}
ZH_VAGUE_WORDS = {
    '东西', '事情', '那个', '这个', '这里', '那里', '什么', '有人', '一些'
}

EN_EXTRA_FILLERS = ['well', 'actually', 'basically', 'so', 'like', 'you know']
ZH_EXTRA_FILLERS = ['嗯', '呃', '额', '那个', '这个', '就是', '然后', '啊']

EN_SELF_REPAIR = [
    'i mean', 'sorry', 'i meant', 'wait', 'hold on', 'hang on',
    'rephrase', 'start again', 'rather'
]
ZH_SELF_REPAIR = [
    '不对', '不是', '我重说', '重新说', '等一下', '等等', '我是说', '更正', '应该是'
]

ZH_PRONOUNS = {
    '我', '你', '他', '她', '它', '我们', '你们', '他们', '她们', '它们', '自己', '人家'
}
ZH_AUX_PARTICLES = {
    '是', '会', '能', '可以', '要', '在', '被', '把', '了', '着', '过', '的', '地', '得'
}
ZH_CLAUSE_MARKERS = ['因为', '所以', '如果', '但是', '然后', '虽然', '尽管', '不过', '而且', '并且']

EN_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
ZH_WORD_RE = re.compile(r'[\u4e00-\u9fffA-Za-z0-9]+')
ZH_SENT_SPLIT_RE = re.compile(r'[。！？!?；;]+')

_NLP_MODELS: dict[str, spacy.language.Language] = {}
_EMBED_MODELS: dict[str, SentenceTransformer | None] = {}

def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num / den)

def _is_zh_token(token: str) -> bool:
    return bool(ZH_WORD_RE.search(token))

def _hanzi_count(token: str) -> int:
    return sum(1 for ch in token if '\u4e00' <= ch <= '\u9fff')

def _zipf_freq(word: str, language: str) -> float:
    try:
        return float(zipf_frequency(word, language))
    except Exception:
        # wordfreq's zh path imports jieba at runtime; fallback to neutral value.
        return 3.0 if language == 'zh' else 0.0

def _load_nlp(language: str):
    if language in _NLP_MODELS:
        return _NLP_MODELS[language]

    model_name = ZH_MODEL if language == 'zh' else EN_MODEL
    fallback_lang = 'zh' if language == 'zh' else 'en'

    try:
        model = spacy.load(model_name)
    except Exception:
        model = spacy.blank(fallback_lang)
        if not model.has_pipe('sentencizer'):
            model.add_pipe('sentencizer')

    _NLP_MODELS[language] = model
    return model

def _load_sentence_model(language: str):
    if language in _EMBED_MODELS:
        return _EMBED_MODELS[language]

    model_name = SENTENCE_MODELS.get(language, SENTENCE_MODELS['en'])
    try:
        model = SentenceTransformer(model_name)
    except Exception:
        model = None

    _EMBED_MODELS[language] = model
    return model

def _load_concreteness(path: Path):
    if not path.exists():
        return {}

    concreteness = {}
    with path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return concreteness

        word_col = reader.fieldnames[0]
        conc_col = reader.fieldnames[1]

        for row in reader:
            word = row.get(word_col, '').strip().lower()
            score = row.get(conc_col, '').strip()
            if not word or not score:
                continue
            try:
                concreteness[word] = float(score)
            except ValueError:
                continue

    return concreteness

CONCRETENESS_DIR = Path(__file__).with_name('concreteness')
CONCRETENESS_EN = _load_concreteness(CONCRETENESS_DIR / 'concreteness_en.csv')
CONCRETENESS_ZH = _load_concreteness(CONCRETENESS_DIR / 'concreteness_zh.csv')

def _split_sentences(text: str, language: str):
    text = (text or '').strip()
    if not text:
        return []

    if language == 'zh':
        parts = [s.strip() for s in ZH_SENT_SPLIT_RE.split(text) if s.strip()]
        if len(parts) >= 2:
            return parts

    doc = _load_nlp(language)(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences if sentences else [text]

def _tokenize_words(text: str, language: str):
    text = text or ''
    if not text:
        return []

    if language == 'zh':
        if jieba is not None:
            tokens = [tok.strip() for tok in jieba.lcut(text) if tok.strip() and _is_zh_token(tok)]
            if tokens:
                return tokens
        return [tok for tok in ZH_WORD_RE.findall(text) if tok.strip()]

    return [tok.lower() for tok in EN_WORD_RE.findall(text)]

def _normalize_text(text: str, language: str):
    text = (text or '').lower()
    if language == 'zh':
        text = re.sub(r'[^\u4e00-\u9fffA-Za-z0-9\s]', ' ', text)
    else:
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def _map_jieba_pos(tag: str) -> str:
    if not tag:
        return 'X'
    if tag.startswith('n'):
        return 'NOUN'
    if tag.startswith('v'):
        return 'VERB'
    if tag.startswith('a'):
        return 'ADJ'
    if tag.startswith('d'):
        return 'ADV'
    if tag.startswith('r'):
        return 'PRON'
    if tag.startswith('p'):
        return 'ADP'
    if tag.startswith('c'):
        return 'CCONJ'
    if tag.startswith('u'):
        return 'PART'
    if tag.startswith('m'):
        return 'NUM'
    return 'X'

def _token_records(text: str, language: str):
    text = text or ''
    if not text:
        return []

    records = []
    doc = _load_nlp(language)(text)

    if language == 'en':
        for token in doc:
            if token.is_space or not token.is_alpha:
                continue
            lemma = (token.lemma_ or token.text).lower()
            records.append({
                'text': token.text,
                'lemma': lemma,
                'pos': token.pos_ or 'X',
                'dep': token.dep_ or '',
            })
        return records

    # zh: prefer spaCy POS if available
    has_usable_pos = any(
        (tok.pos_ and tok.pos_ not in {'', 'X', 'SPACE', 'PUNCT'})
        for tok in doc if not tok.is_space
    )

    if has_usable_pos:
        for token in doc:
            if token.is_space:
                continue
            token_text = token.text.strip()
            if not token_text or not _is_zh_token(token_text):
                continue
            lemma = (token.lemma_ or token_text).lower()
            records.append({
                'text': token_text,
                'lemma': lemma,
                'pos': token.pos_ or 'X',
                'dep': token.dep_ or '',
            })
        return records

    if jieba_posseg is not None:
        for word, tag in jieba_posseg.lcut(text):
            word = word.strip()
            if not word or not _is_zh_token(word):
                continue
            records.append({
                'text': word,
                'lemma': word.lower(),
                'pos': _map_jieba_pos(tag),
                'dep': '',
            })
        return records

    for token in _tokenize_words(text, language='zh'):
        pos = 'X'
        if token in ZH_PRONOUNS:
            pos = 'PRON'
        elif token in ZH_AUX_PARTICLES:
            pos = 'PART'
        records.append({
            'text': token,
            'lemma': token.lower(),
            'pos': pos,
            'dep': '',
        })
    return records

def _concreteness_map(language: str):
    return CONCRETENESS_ZH if language == 'zh' else CONCRETENESS_EN

def _text_stats(transcript: dict, language: str):
    text = transcript.get('text', '')
    words = _tokenize_words(text, language)
    total_tokens = len(words)
    unique_tokens = len(set(words))
    type_token_ratio = _safe_div(unique_tokens, total_tokens)

    utterances = [
        len(_tokenize_words(seg.get('text', ''), language))
        for seg in transcript.get('segments', [])
    ]

    if utterances:
        mean_words_per_utterance = float(np.mean(utterances))
        max_utterance_length = max(utterances)
    else:
        mean_words_per_utterance = 0.0
        max_utterance_length = 0

    sentence_count = len(_split_sentences(text, language))

    return (
        total_tokens,
        unique_tokens,
        type_token_ratio,
        mean_words_per_utterance,
        max_utterance_length,
        sentence_count,
    )

def _lexical_richness(transcript: dict, language: str):
    text = transcript.get('text', '')
    records = _token_records(text, language)

    total_words = len(records)
    content_words = 0
    function_words = 0
    rare_words = 0

    vague_words = ZH_VAGUE_WORDS if language == 'zh' else EN_VAGUE_WORDS

    for token in records:
        lemma = token['lemma']
        pos = token['pos']

        if pos in CONTENT_POS and lemma not in vague_words:
            content_words += 1
        elif pos in FUNCTION_POS:
            function_words += 1

        freq = _zipf_freq(lemma, language)
        if freq < 3.0:
            rare_words += 1

    content_words_ratio = _safe_div(content_words, total_words)
    function_words_ratio = _safe_div(function_words, total_words)
    rare_words_ratio = _safe_div(rare_words, total_words)

    return content_words_ratio, function_words_ratio, rare_words_ratio

def _repetition_disfluency(transcript: dict, language: str):
    text = transcript.get('text', '')
    normalized = _normalize_text(text, language)

    fillers = transcript.get('fillers', 0)
    extra_fillers = ZH_EXTRA_FILLERS if language == 'zh' else EN_EXTRA_FILLERS
    filler_count = fillers + sum(normalized.count(item) for item in extra_fillers)

    self_repair_markers = ZH_SELF_REPAIR if language == 'zh' else EN_SELF_REPAIR
    self_correction_count = sum(normalized.count(item) for item in self_repair_markers)

    tokens = _tokenize_words(text, language)
    total_tokens = len(tokens)

    repetition_score = 0.0
    bigram_repetition_ratio = 0.0

    for n in range(1, 6):
        if total_tokens < n:
            continue

        ngrams = [' '.join(tokens[i:i + n]) for i in range(total_tokens - n + 1)]
        counts = Counter(ngrams)
        repeated = {ngram: count for ngram, count in counts.items() if count > 1}

        if n == 2:
            raw_repeat = sum(count - 1 for count in repeated.values())
            bigram_repetition_ratio = _safe_div(raw_repeat, len(ngrams))

        repetition_score += sum((count - 1) * n for count in repeated.values())

    repetition_score = _safe_div(repetition_score, total_tokens)

    return filler_count, repetition_score, bigram_repetition_ratio, self_correction_count

def _encode_sentences(sentences: list[str], language: str):
    model = _load_sentence_model(language)
    if model is None:
        return None

    try:
        return model.encode(sentences)
    except Exception:
        return None

def _semantic_coherence(transcript: dict, language: str):
    text = transcript.get('text', '')
    sentences = [_normalize_text(s, language) for s in _split_sentences(text, language) if s.strip()]

    if len(sentences) < 2:
        return 0.0, 0.0

    embeddings = _encode_sentences(sentences, language)
    if embeddings is None:
        return 0.0, 0.0

    similarities = []
    for i in range(len(embeddings) - 1):
        v1 = embeddings[i]
        v2 = embeddings[i + 1]
        denom = norm(v1) * norm(v2)
        sim = 0.0 if denom == 0 else float(np.dot(v1, v2) / denom)
        similarities.append(sim)

    if not similarities:
        return 0.0, 0.0

    return float(np.mean(similarities)), float(np.var(similarities))

def _syntactic_complexity_parser(text: str, language: str):
    nlp = _load_nlp(language)
    if not nlp.has_pipe('parser'):
        return None

    doc = nlp(text)

    dependency_distances = []
    clause_count = 0
    sentence_count = 0

    clause_deps = {'ccomp', 'xcomp', 'advcl', 'acl', 'relcl'}
    if language == 'zh':
        clause_deps = clause_deps | {'conj', 'parataxis'}

    for sent in doc.sents:
        sentence_count += 1
        clause_count += 1
        for token in sent:
            if token.head != token:
                dependency_distances.append(abs(token.i - token.head.i))
            if token.dep_ in clause_deps:
                clause_count += 1

    mean_dependency_distance = float(np.mean(dependency_distances)) if dependency_distances else 0.0
    clause_density = _safe_div(clause_count, sentence_count)

    def tree_height(token):
        children = list(token.children)
        if not children:
            return 1
        return 1 + max(tree_height(child) for child in children)

    tree_heights = [tree_height(sent.root) for sent in doc.sents]
    mean_tree_height = float(np.mean(tree_heights)) if tree_heights else 0.0

    return mean_dependency_distance, clause_density, mean_tree_height

def _syntactic_complexity_heuristic(text: str, language: str):
    sentences = _split_sentences(text, language)
    if not sentences:
        return 0.0, 0.0, 0.0

    tokenized = [_tokenize_words(sentence, language) for sentence in sentences]

    dep_proxy = [max(0.0, (len(tokens) - 1) / 2.0) for tokens in tokenized if tokens]
    mean_dependency_distance = float(np.mean(dep_proxy)) if dep_proxy else 0.0

    clause_count = len(sentences)
    if language == 'zh':
        clause_count += sum(text.count(marker) for marker in ZH_CLAUSE_MARKERS)
    else:
        clause_count += sum(text.lower().count(marker) for marker in ['because', 'that', 'which', 'who', 'if'])

    clause_density = _safe_div(clause_count, len(sentences))

    heights = []
    for tokens in tokenized:
        n = len(tokens)
        if n == 0:
            continue
        heights.append(max(1, int(math.ceil(math.log2(n + 1)))))

    mean_tree_height = float(np.mean(heights)) if heights else 0.0

    return mean_dependency_distance, clause_density, mean_tree_height

def _syntactic_complexity(transcript: dict, language: str):
    text = transcript.get('text', '')

    parsed = _syntactic_complexity_parser(text, language)
    if parsed is not None:
        return parsed

    return _syntactic_complexity_heuristic(text, language)

def _pos_ratios(transcript: dict, language: str):
    records = _token_records(transcript.get('text', ''), language)

    total_words = len(records)
    pronoun_count = 0
    noun_count = 0
    verb_count = 0
    aux_count = 0

    for token in records:
        word = token['text']
        pos = token['pos']

        if pos == 'PRON' or (language == 'zh' and word in ZH_PRONOUNS):
            pronoun_count += 1
        if pos in {'NOUN', 'PROPN'}:
            noun_count += 1
        if pos == 'VERB':
            verb_count += 1
        if pos in {'AUX', 'PART'} or (language == 'zh' and word in ZH_AUX_PARTICLES):
            aux_count += 1

    pronoun_ratio = _safe_div(pronoun_count, total_words)
    verb_to_noun_ratio = _safe_div(verb_count, noun_count)
    auxiliary_verb_ratio = _safe_div(aux_count, total_words)

    return pronoun_ratio, verb_to_noun_ratio, auxiliary_verb_ratio

def _semantic_content(transcript: dict, language: str):
    records = _token_records(transcript.get('text', ''), language)
    word_count = len(records)

    proposition_pos = {'VERB', 'ADJ', 'ADV', 'ADP', 'CCONJ', 'SCONJ'}
    if language == 'zh':
        proposition_pos = proposition_pos | {'PART'}

    proposition_count = sum(1 for token in records if token['pos'] in proposition_pos)
    idea_density = _safe_div(proposition_count, word_count)

    concreteness_map = _concreteness_map(language)
    concreteness_scores = []

    for token in records:
        if token['pos'] not in {'NOUN', 'PROPN', 'VERB', 'ADJ'}:
            continue
        key = token['lemma'].lower()
        score = concreteness_map.get(key)
        if score is not None:
            concreteness_scores.append(score)

    mean_concreteness = float(np.mean(concreteness_scores)) if concreteness_scores else 0.0

    abstract_count = 0
    for token in records:
        key = token['lemma'].lower()
        score = concreteness_map.get(key)
        if score is not None and score < 2.0:
            abstract_count += 1

    abstract_ratio = _safe_div(abstract_count, word_count)

    return idea_density, mean_concreteness, abstract_ratio

def _zh_syllables(token: str) -> int:
    if not token:
        return 0

    if lazy_pinyin is not None:
        try:
            py = [item for item in lazy_pinyin(token) if item.strip()]
            if py:
                return len(py)
        except Exception:
            pass

    hanzi = _hanzi_count(token)
    if hanzi > 0:
        return hanzi

    ascii_parts = re.findall(r'[aeiouy]+', token.lower())
    return max(1, len(ascii_parts)) if token else 0

def _vocabulary_sophistication(transcript: dict, language: str):
    text = transcript.get('text', '')

    if language == 'en':
        fk_grade = textstat.flesch_kincaid_grade(text) if text else 0.0
        words = _tokenize_words(text, language)
        syllables = [textstat.syllable_count(word) for word in words]
        mean_syllables = float(np.mean(syllables)) if syllables else 0.0
        long_words = sum(1 for word in words if len(word) > 6)
        long_word_ratio = _safe_div(long_words, len(words))
        return fk_grade, mean_syllables, long_word_ratio

    tokens = _tokenize_words(text, language='zh')
    if not tokens:
        return 0.0, 0.0, 0.0

    sentences = _split_sentences(text, language='zh')
    sentence_lengths = [len(_tokenize_words(sentence, language='zh')) for sentence in sentences if sentence.strip()]
    avg_sentence_len = float(np.mean(sentence_lengths)) if sentence_lengths else float(len(tokens))

    rare_ratio = _safe_div(sum(1 for token in tokens if _zipf_freq(token, 'zh') < 3.0), len(tokens))
    readability_proxy = avg_sentence_len + (4.0 * rare_ratio)

    syllable_counts = [_zh_syllables(token) for token in tokens]
    mean_syllables = float(np.mean(syllable_counts)) if syllable_counts else 0.0

    long_tokens = sum(1 for token in tokens if _hanzi_count(token) >= 3)
    long_word_ratio = _safe_div(long_tokens, len(tokens))

    return readability_proxy, mean_syllables, long_word_ratio

def _tfidf_tokenizer_zh(text: str):
    return _tokenize_words(text, language='zh')

def _discourse_coherence(transcript: dict, language: str):
    sentences = _split_sentences(transcript.get('text', ''), language)

    if len(sentences) < 3:
        return 0.0, 0.0

    embeddings = _encode_sentences(sentences, language)
    if embeddings is None:
        return 0.0, 0.0

    first = embeddings[0]
    similarities_to_first = []

    for i in range(1, len(embeddings)):
        current = embeddings[i]
        denom = norm(first) * norm(current)
        sim = 0.0 if denom == 0 else float(np.dot(first, current) / denom)
        similarities_to_first.append(sim)

    if len(similarities_to_first) > 1:
        x = np.arange(len(similarities_to_first))
        slope = np.polyfit(x, similarities_to_first, 1)[0]
        global_coherence_drift = float(-slope)
    else:
        global_coherence_drift = 0.0

    try:
        if language == 'zh':
            vectorizer = TfidfVectorizer(
                max_features=10,
                tokenizer=_tfidf_tokenizer_zh,
                token_pattern=None
            )
        else:
            vectorizer = TfidfVectorizer(max_features=10, stop_words='english')

        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()

        topic_counts = Counter()
        for i in range(tfidf_matrix.shape[0]):
            row = tfidf_matrix[i].toarray()[0]
            if row.size and row.max() > 0:
                top_topic_idx = row.argmax()
                topic_counts[feature_names[top_topic_idx]] += 1

        total_occurrences = sum(topic_counts.values())
        recurrences = sum(count - 1 for count in topic_counts.values() if count > 1)
        topic_recurrence = _safe_div(recurrences, total_occurrences)
    except Exception:
        topic_recurrence = 0.0

    return global_coherence_drift, topic_recurrence

def extract(fn: Path, transcript: dict, use_cache=False, verbose=False, language='en'):
    lang = normalize_language(language or transcript.get('language'))

    linguistics = None
    cache_variant = None if lang == 'en' else f'ling:{lang}'
    cache_file = cache.key(fn, CACHE_DIR, variant=cache_variant)
    if use_cache:
        linguistics = cache.load(cache_file)

    if linguistics is None:
        if verbose:
            print(f'[LING] Extracting linguistic features ({lang})')

        (
            total_tokens,
            unique_tokens,
            type_token_ratio,
            mean_words_per_utterance,
            max_utterance_length,
            sentence_count,
        ) = _text_stats(transcript, lang)

        content_words_ratio, function_words_ratio, rare_words_ratio = _lexical_richness(transcript, lang)
        filler_count, repetition_score, bigram_repetition_ratio, self_correction_count = _repetition_disfluency(
            transcript, lang
        )
        semantic_coherence_mean, semantic_coherence_variance = _semantic_coherence(transcript, lang)
        mean_dependency_distance, clause_density, mean_parse_tree_height = _syntactic_complexity(transcript, lang)
        pronoun_ratio, verb_to_noun_ratio, auxiliary_verb_ratio = _pos_ratios(transcript, lang)
        idea_density, mean_concreteness, abstract_ratio = _semantic_content(transcript, lang)
        fk_grade, mean_syllables, long_word_ratio = _vocabulary_sophistication(transcript, lang)
        global_coherence_drift, topic_recurrence = _discourse_coherence(transcript, lang)

        feature_vector = np.array([
            # Basic text stats (6)
            total_tokens,
            unique_tokens,
            type_token_ratio,
            mean_words_per_utterance,
            max_utterance_length,
            sentence_count,

            # Lexical richness (3)
            content_words_ratio,
            function_words_ratio,
            rare_words_ratio,

            # Repetition & disfluency (4)
            filler_count,
            repetition_score,
            bigram_repetition_ratio,
            self_correction_count,

            # Semantic coherence (2)
            semantic_coherence_mean,
            semantic_coherence_variance,

            # Syntactic complexity (3)
            mean_dependency_distance,
            clause_density,
            mean_parse_tree_height,

            # Parts-of-speech ratios (3)
            pronoun_ratio,
            verb_to_noun_ratio,
            auxiliary_verb_ratio,

            # Semantic content (3)
            idea_density,
            mean_concreteness,
            abstract_ratio,

            # Vocabulary sophistication (3)
            fk_grade,
            mean_syllables,
            long_word_ratio,

            # Discourse coherence (2)
            global_coherence_drift,
            topic_recurrence,
        ])

        if verbose:
            print('[LING] Done extracting')

        linguistics = np.nan_to_num(
            feature_vector,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        cache.save(cache_file, linguistics)

    return linguistics
