# Feature Schema

This document defines the ordered inputs produced by the Neurolens feature
extractors. Feature order is part of the model contract: changing it requires
regenerating caches, preprocessing artifacts, feature arrays, and weights.

## Dimensions

| Block | Extracted dimensions | Model dimensions |
|---|---:|---:|
| Acoustic | 52 | 52 |
| Linguistic | 29 | 29 |
| Semantic | 18 | 18 |
| HuBERT | 1024 | 128 after standardization and PCA |
| **Total** | **1123** | **227** |

The 99 acoustic, linguistic, and semantic values are standardized together.
The HuBERT block is standardized separately and projected to 128 principal
components. Both scalers and PCA are fitted on training data only.

## Acoustic Features

Implementation: `features/acoustics.py`

Audio is loaded at 16 kHz. WebRTC VAD with aggressiveness level 3 identifies
voiced 30 ms frames. A pause is counted after at least eight consecutive
non-speech frames, corresponding to approximately 240 ms.

| Index | Feature | Definition |
|---:|---|---|
| 1 | Mean F0 | Mean YIN fundamental frequency over VAD-retained audio |
| 2 | F0 standard deviation | Standard deviation of fundamental frequency |
| 3 | Minimum F0 | Minimum detected fundamental frequency |
| 4 | Maximum F0 | Maximum detected fundamental frequency |
| 5 | F0 interquartile range | 75th minus 25th percentile of F0 |
| 6 | Mean RMS energy | Mean frame-wise root-mean-square energy |
| 7 | RMS energy standard deviation | Variability of frame-wise RMS energy |
| 8 | Energy dynamic range | Maximum minus minimum frame-wise RMS energy |
| 9 | Words per second | Regex token count divided by VAD-retained duration |
| 10 | Syllables per second | Heuristic English syllable count divided by VAD-retained duration |
| 11 | Pause count | Number of non-speech runs lasting at least about 240 ms |
| 12 | Total pause duration | Full duration minus VAD-retained duration |
| 13 | Pause ratio | Total pause duration divided by full duration |
| 14 | MFCC 1 mean | Mean of MFCC coefficient 1 |
| 15 | MFCC 1 standard deviation | Standard deviation of MFCC coefficient 1 |
| 16 | MFCC 2 mean | Mean of MFCC coefficient 2 |
| 17 | MFCC 2 standard deviation | Standard deviation of MFCC coefficient 2 |
| 18 | MFCC 3 mean | Mean of MFCC coefficient 3 |
| 19 | MFCC 3 standard deviation | Standard deviation of MFCC coefficient 3 |
| 20 | MFCC 4 mean | Mean of MFCC coefficient 4 |
| 21 | MFCC 4 standard deviation | Standard deviation of MFCC coefficient 4 |
| 22 | MFCC 5 mean | Mean of MFCC coefficient 5 |
| 23 | MFCC 5 standard deviation | Standard deviation of MFCC coefficient 5 |
| 24 | MFCC 6 mean | Mean of MFCC coefficient 6 |
| 25 | MFCC 6 standard deviation | Standard deviation of MFCC coefficient 6 |
| 26 | MFCC 7 mean | Mean of MFCC coefficient 7 |
| 27 | MFCC 7 standard deviation | Standard deviation of MFCC coefficient 7 |
| 28 | MFCC 8 mean | Mean of MFCC coefficient 8 |
| 29 | MFCC 8 standard deviation | Standard deviation of MFCC coefficient 8 |
| 30 | MFCC 9 mean | Mean of MFCC coefficient 9 |
| 31 | MFCC 9 standard deviation | Standard deviation of MFCC coefficient 9 |
| 32 | MFCC 10 mean | Mean of MFCC coefficient 10 |
| 33 | MFCC 10 standard deviation | Standard deviation of MFCC coefficient 10 |
| 34 | MFCC 11 mean | Mean of MFCC coefficient 11 |
| 35 | MFCC 11 standard deviation | Standard deviation of MFCC coefficient 11 |
| 36 | MFCC 12 mean | Mean of MFCC coefficient 12 |
| 37 | MFCC 12 standard deviation | Standard deviation of MFCC coefficient 12 |
| 38 | MFCC 13 mean | Mean of MFCC coefficient 13 |
| 39 | MFCC 13 standard deviation | Standard deviation of MFCC coefficient 13 |
| 40 | Spectral-centroid mean | Mean spectral centroid |
| 41 | Spectral-centroid standard deviation | Variability of spectral centroid |
| 42 | Spectral-bandwidth mean | Mean spectral bandwidth |
| 43 | Spectral-bandwidth standard deviation | Variability of spectral bandwidth |
| 44 | Spectral-flux mean | Mean onset-strength value |
| 45 | Spectral-flux standard deviation | Variability of onset strength |
| 46 | Spectral slope | Linear slope of the mean magnitude spectrum |
| 47 | Jitter | Praat local period-to-period F0 variation |
| 48 | Shimmer | Praat local period-to-period amplitude variation |
| 49 | Harmonics-to-noise ratio | Mean Praat harmonicity |
| 50 | Cepstral peak prominence | Praat CPPS |
| 51 | Zero-crossing-rate mean | Mean frame-wise zero-crossing rate |
| 52 | Zero-crossing-rate standard deviation | Variability of zero-crossing rate |

Important implementation detail: indices 9 and 10 are **words per second**
followed by **syllables per second**. Older documentation listed these in the
opposite order.

The acoustic syllable heuristic strips non-ASCII letters and counts English
vowel groups. It is not Mandarin-aware, even when the transcript language is
`zh`.

## Linguistic Features

Implementation: `features/linguistics.py`

| Index | Feature | Definition |
|---:|---|---|
| 1 | Total tokens | Number of language-specific word tokens |
| 2 | Unique tokens | Number of unique tokens |
| 3 | Type-token ratio | Unique tokens divided by total tokens |
| 4 | Mean words per utterance | Mean token count across Whisper segments |
| 5 | Maximum utterance length | Largest segment token count |
| 6 | Sentence count | Number of detected sentence units |
| 7 | Content-word ratio | Content POS tokens divided by parsed tokens |
| 8 | Function-word ratio | Function POS tokens divided by parsed tokens |
| 9 | Rare-word ratio | Tokens with wordfreq Zipf frequency below 3 |
| 10 | Filler count | ASR filler count plus configured lexical fillers |
| 11 | Repetition score | Length-weighted repeated 1- to 5-gram count per token |
| 12 | Bigram repetition ratio | Repeated bigram occurrences divided by all bigrams |
| 13 | Self-correction count | Count of language-specific repair markers |
| 14 | Mean local coherence | Mean cosine similarity of adjacent sentence embeddings |
| 15 | Local-coherence variance | Variance of adjacent-sentence cosine similarities |
| 16 | Mean dependency distance | Mean token distance to syntactic head, or heuristic proxy |
| 17 | Clause density | Estimated clauses per sentence |
| 18 | Mean parse-tree height | Mean dependency-tree height, or heuristic proxy |
| 19 | Pronoun ratio | Pronouns divided by parsed tokens |
| 20 | Verb-to-noun ratio | Verbs divided by nouns and proper nouns |
| 21 | Auxiliary/particle ratio | Auxiliaries and particles divided by parsed tokens |
| 22 | Idea density | Proposition-bearing POS tokens divided by parsed tokens |
| 23 | Mean concreteness | Mean matched concreteness score |
| 24 | Abstract-word ratio | Matched words with concreteness below 2, divided by tokens |
| 25 | Readability | Flesch-Kincaid grade for English; Mandarin proxy otherwise |
| 26 | Mean syllables | Mean textstat syllables for English; pinyin/Hanzi count for Mandarin |
| 27 | Long-word ratio | English tokens longer than six characters; Mandarin tokens with at least three Hanzi |
| 28 | Global coherence drift | Negative slope of similarity to the first sentence |
| 29 | Topic recurrence | Repeated top TF-IDF topics divided by sentence-topic assignments |

### Language-specific behavior

English uses:

- spaCy `en_core_web_sm`, with `spacy.blank("en")` fallback;
- `all-mpnet-base-v2` sentence embeddings;
- `textstat` Flesch-Kincaid grade and syllable counts; and
- English word-frequency and concreteness resources.

Mandarin uses:

- spaCy `zh_core_web_sm` when available;
- jieba tokenization/POS fallback, then simpler heuristics;
- `paraphrase-multilingual-mpnet-base-v2` sentence embeddings;
- pinyin or Hanzi-based syllable counts;
- a custom readability proxy based on sentence length and rare-word ratio; and
- Mandarin-specific filler, repair, concreteness, and clause-marker resources.

If a sentence-transformer model cannot load or encode text, coherence features
fall back to zero. If a spaCy parser is unavailable, syntactic features use
heuristics. These fallbacks preserve vector shape but not feature equivalence.

## Semantic Features

Implementation: `features/semantics.py`

The semantic extractor asks a local Ollama model to assign one score per
rubric. Scores are expected in the range 0-4. English and Mandarin use
language-specific feature-list JSON files, but retain the same feature order.

| Index | Feature | Construct represented by the rubric |
|---:|---|---|
| 1 | Semantic memory degradation | Fact loss, retrieval failure, or memory-related semantic errors |
| 2 | Narrative structure disintegration | Temporal, causal, or story-structure breakdown |
| 3 | Pragmatic appropriateness | Fit between the response and conversational intent |
| 4 | Topic maintenance | Ability to remain on the elicited topic |
| 5 | Perseveration types | Recurrent or intrusive repetition patterns |
| 6 | Disorientation types | Temporal, spatial, or personal confusion |
| 7 | Executive dysfunction patterns | Planning, initiation, organization, or response-control difficulty |
| 8 | Abstract reasoning | Ability to generalize beyond literal or concrete content |
| 9 | Semantic clustering vs. fragmentation | Thematic organization versus disconnected ideas |
| 10 | Emotional appropriateness | Fit between affective tone and content |
| 11 | Novel information content | Production of new, meaningful information |
| 12 | Ambiguity and vagueness | Reliance on underspecified or empty references |
| 13 | Instruction following | Whether the answer addresses the elicitation request |
| 14 | Logical self-consistency | Internal contradiction or inconsistency |
| 15 | Confabulation | Unsupported but plausible invented content |
| 16 | Clinical impression | Rubric-based overall impairment impression |
| 17 | Error-type classification | Semantic, retrieval, syntactic, phonological, or executive errors |
| 18 | Compensation strategies | Circumlocution, avoidance, or memory-related meta-commentary |

The implementation requests scores in five sections. Invalid individual scores
are replaced with `1.0`. Batch extraction retries parse failures and ultimately
uses a full vector of `1.0` values after repeated failure. The configured model
is `ministral-3:8b`, with temperature 0.

These scores are model-generated annotations, not clinician ratings.

## HuBERT Representation

Implementation: `processing/transcriber.py`

`facebook/hubert-large-ll60k` produces a sequence of 1024-dimensional hidden
states. Neurolens mean-pools over the time axis to obtain one
1024-dimensional vector per recording.

Current training then:

1. fits a `StandardScaler` on training HuBERT vectors;
2. transforms train, validation, and test vectors;
3. fits `PCA(n_components=128)` on the scaled training vectors; and
4. concatenates the 128 PCA components after the 99 standardized
   handcrafted/semantic features.

PCA components are learned features and do not have stable semantic names.

## Final Input Order

The current model receives:

```text
0:52      acoustic features
52:81     linguistic features
81:99     semantic features
99:227    PCA-reduced HuBERT components
```

The raw extraction order before preprocessing is:

```text
0:52       acoustic features
52:81      linguistic features
81:99      semantic features
99:1123    raw HuBERT dimensions
```

Tracked repository checkpoints and `model_scaler.pkl` currently correspond to
the older raw 1123-dimensional path. They are not compatible with the current
227-dimensional default architecture.
