# Neurolens AI Feature Inventory

This document lists **all feature types** extracted by this stage of the Neurolens pipeline:  
- Acoustic features (`acoustics.py`)  
- Linguistic features (`linguistics.py`)  
- LLM-derived semantic scores (`semantics.py`)
- 
---

## 🎧 Acoustic Features (42)

| #  | Category              | Feature Name                 | Description                   |
|----|-----------------------|------------------------------|-------------------------------|
| 1  | Pitch                 | Mean F0                      | Average fundamental frequency |
| 2  | Pitch                 | STD F0                       | Variability of pitch          |
| 3  | Pitch                 | Min F0                       | Lowest detected pitch         |
| 4  | Pitch                 | Max F0                       | Highest detected pitch        |
| 5  | Energy                | Mean energy                  | Average loudness              |
| 6  | Energy                | STD energy                   | Variability in loudness       |
| 7  | Energy                | Dynamic range                | Max energy − Min energy       |
| 8  | Speaking Rate         | Syllables/sec                | Estimated speech speed        |
| 9  | Speaking Rate         | Words/sec                    | Transcript-aligned speed      |
| 10 | Pauses                | # pauses > X ms              | Count of long pauses          |
| 11 | Pauses                | Total pause duration         | Sum of all pauses             |
| 12 | Pauses                | Pause ratio                  | pause_time / total_time       |
| 13 | MFCCs (1–13)          | MFCC1 mean                   | Mean of coefficient 1         |
| 14 | MFCCs (1–13)          | MFCC1 std                    | Std of coefficient 1          |
| 15 | MFCCs (1–13)          | MFCC2 mean                   | Mean of coefficient 2         |
| 16 | MFCCs (1–13)          | MFCC2 std                    | Std of coefficient 2          |
| 17 | MFCCs (1–13)          | MFCC3 mean                   | Mean of coefficient 3         |
| 18 | MFCCs (1–13)          | MFCC3 std                    | Std of coefficient 3          |
| 19 | MFCCs (1–13)          | MFCC4 mean                   | Mean of coefficient 4         |
| 20 | MFCCs (1–13)          | MFCC4 std                    | Std of coefficient 4          |
| 21 | MFCCs (1–13)          | MFCC5 mean                   | Mean of coefficient 5         |
| 22 | MFCCs (1–13)          | MFCC5 std                    | Std of coefficient 5          |
| 23 | MFCCs (1–13)          | MFCC6 mean                   | Mean of coefficient 6         |
| 24 | MFCCs (1–13)          | MFCC6 std                    | Std of coefficient 6          |
| 25 | MFCCs (1–13)          | MFCC7 mean                   | Mean of coefficient 7         |
| 26 | MFCCs (1–13)          | MFCC7 std                    | Std of coefficient 7          |
| 27 | MFCCs (1–13)          | MFCC8 mean                   | Mean of coefficient 8         |
| 28 | MFCCs (1–13)          | MFCC8 std                    | Std of coefficient 8          |
| 29 | MFCCs (1–13)          | MFCC9 mean                   | Mean of coefficient 9         |
| 30 | MFCCs (1–13)          | MFCC9 std                    | Std of coefficient 9          |
| 31 | MFCCs (1–13)          | MFCC10 mean                  | Mean of coefficient 10        |
| 32 | MFCCs (1–13)          | MFCC10 std                   | Std of coefficient 10         |
| 33 | MFCCs (1–13)          | MFCC11 mean                  | Mean of coefficient 11        |
| 34 | MFCCs (1–13)          | MFCC11 std                   | Std of coefficient 11         |
| 35 | MFCCs (1–13)          | MFCC12 mean                  | Mean of coefficient 12        |
| 36 | MFCCs (1–13)          | MFCC12 std                   | Std of coefficient 12         |
| 37 | MFCCs (1–13)          | MFCC13 mean                  | Mean of coefficient 13        |
| 38 | MFCCs (1–13)          | MFCC13 std                   | Std of coefficient 13         |
| 39 | Spectral Centroid     | Mean spectral centroid       | Brightness of sound           |
| 40 | Spectral Centroid     | STD spectral centroid        | Variability in brightness     |
| 41 | Spectral Bandwidth    | Mean spectral bandwidth      | Spread of frequencies         |
| 42 | Spectral Bandwidth    | STD spectral bandwidth       | Variability in spread         |

---

## ✍️ Linguistic Features (15)

| #  | Category                 | Feature Name             | Description                                  |
|----|--------------------------|--------------------------|----------------------------------------------|
| 1  | Basic Text Stats         | Total tokens             | Total number of tokens in transcript         |
| 2  | Basic Text Stats         | Unique tokens            | Number of unique vocabulary items            |
| 3  | Basic Text Stats         | Type–token ratio         | unique / total tokens                        |
| 4  | Basic Text Stats         | Mean words per utterance | Avg. words per Whisper segment               |
| 5  | Basic Text Stats         | Max utterance length     | Longest utterance in words                   |
| 6  | Basic Text Stats         | Number of sentences      | Approx. sentence count                       |
| 7  | Lexical Richness         | Content-word ratio       | (nouns + verbs + adj + adv) / total          |
| 8  | Lexical Richness         | Function-word ratio      | function words / total                       |
| 9  | Lexical Richness         | Rare-word ratio          | rare words / total                           |
| 10 | Repetition & Disfluency  | Filler-word count        | Count of {"um","uh","like","you know","er"}  |
| 11 | Repetition & Disfluency  | Repetition score         | Weighted count of repeated words/phrases     |
| 12 | Repetition & Disfluency  | Bigram repetition ratio  | repeated bigrams / total bigrams             |
| 13 | Repetition & Disfluency  | Self-correction count    | Count of {"sorry","I mean","no wait"}        |
| 14 | Semantic Coherence       | Mean local coherence     | Mean cosine similarity between segments      |
| 15 | Semantic Coherence       | Coherence variance       | Variance of consecutive-segment similarities |

---

## 🧠 LLM Semantic Scores (18)

| #  | Category                         | Feature Name                         | Description                                                           |
|----|----------------------------------|--------------------------------------|-----------------------------------------------------------------------|
| 1  | Semantic Understanding           | Semantic memory degradation          | Mis-remembering facts, confabulation, memory-failure signals          |
| 2  | Semantic Understanding           | Narrative structure disintegration   | Temporal disorder, missing causal links, broken storyline             |
| 3  | Semantic Understanding           | Pragmatic appropriateness            | Answer fits the question intent, not over/under-explained             |
| 4  | Semantic Understanding           | Topic maintenance                    | Rate at which the speaker drifts off-topic                            |
| 5  | Clinical Signature Behaviors     | Perseveration types                  | Stuck loops, intrusive repetition, returning to old ideas             |
| 6  | Clinical Signature Behaviors     | Disorientation types                 | Temporal, spatial, or personal confusion                              |
| 7  | Clinical Signature Behaviors     | Executive dysfunction patterns       | Not answering questions, filler responses, inability to plan          |
| 8  | Clinical Signature Behaviors     | Abstract reasoning                   | Inability to generalize, overly literal answers                       |
| 9  | Discourse-Level Signals          | Semantic clustering vs fragmentation | Sentences cluster into themes vs fragmented ideas                     |
| 10 | Discourse-Level Signals          | Emotional appropriateness            | Emotional tone matches or mismatches content                          |
| 11 | Discourse-Level Signals          | Novel information content            | Amount of new meaningful information per unit speech                  |
| 12 | Discourse-Level Signals          | Ambiguity & vagueness                | Overuse of vague/empty expressions (“that thing”, “the place”)        |
| 13 | Q/A Relationship                 | Instruction following                | Whether the answer addresses the actual question                      |
| 14 | Q/A Relationship                 | Logical self-consistency             | Contradictions within an answer                                       |
| 15 | Q/A Relationship                 | Confabulation                        | Plausible but fabricated details or stories                           |
| 16 | Meta-Features                    | Clinical impression                  | LLM-estimated cognitive impairment severity                           |
| 17 | Meta-Features                    | Error type classification            | Semantic, retrieval, syntactic, phonological, or executive errors     |
| 18 | Meta-Features                    | Compensation strategies              | Circumlocution, avoidance, meta-comments about memory                 |