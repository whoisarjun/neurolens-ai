# 🧠 Neurolens AI – Full Cognitive Assessment Pipeline

A fully modern, multimodal, clinically-aligned speech‑based cognitive assessment system.

---

## **Pipeline Overview (Stage Names Only)**

1.  **Session Ingestion & Metadata Capture**
2.  **Audio Preprocessing & ASR with Timestamps**
3.  **Multimodal Feature Extraction**
4.  **Session-Level Fusion over 5 Questions**
5.  **Longitudinal Modeling Across Sessions**
6.  **Multi-Task Prediction (Diagnosis, Severity, Risk Index)**
7.  **Uncertainty, Quality Gating & Safety Checks**
8.  **Explanation & Clinician-Facing Report Generation**

---

# **1️⃣ Session Ingestion & Metadata Capture**

### **What comes in**

Each session contains five question–answer pairs:

-   **Question text**
-   **Question type** (picture description, memory, orientation, fluency, executive)
-   **Raw answer audio**
-   **Transcript** (after ASR)
-   **Metadata per patient:**
    -   Age, sex
    -   Education level
    -   Primary language/dialect
    -   Diagnoses (SMC, MCI, AD)
    -   Comorbidities (hearing loss, depression)
-   **Session metadata:**
    -   Device type
    -   Environment flags
    -   Timestamp

### **What comes out**

A structured JSON-like session object:

```json
{
  "patient_id": "...",
  "session_id": "...",
  "timestamp": "...",
  "demographics": {...},
  "clinical_context": {...},
  "interactions": [
    {
      "q_id": "...",
      "question_text": "...",
      "question_type": "...",
      "audio_path": "...",
      "raw_transcript": null,
      "call_metadata": {...}
    }
  ]
}
```

### **Why it matters**

-   Ensures consistent structure → fewer bugs.
-   Controls which signals the model can use (e.g., education-adjusted norms).
-   Avoids hidden data leakage.

---

# **2️⃣ Audio Preprocessing & ASR with Timestamps**

### **Steps**

-   **VAD:** remove long silences
-   **Diarization:** isolate patient’s voice
-   **Noise reduction + normalization**
-   **ASR with timestamps:**
    -   Transcript
    -   Word-level timing
    -   Confidence

### **Quality measures**

-   Speech duration
-   SNR estimate
-   ASR perplexity/confidence

### **Why it matters**

If this stage is bad → the whole pipeline collapses.  
Everything downstream depends on clean audio + clean timestamps.

---

# **3️⃣ Multimodal Feature Extraction**

Transforms each answer into a dense feature vector.

---

## **3.1 Acoustic Features**

-   Speech rate, articulation rate
-   Pause counts + durations
-   Ratio of speech time / total time
-   Pitch mean + variability
-   Jitter/shimmer‑like stats
-   **SSL audio embeddings** (Whisper encoder, wav2vec2, etc.)

**Purpose:** captures *how* they speak, independent of text.

---

## **3.2 Linguistic Features**

-   Lexical richness (type‑token ratio)
-   Syntax (POS distribution, complexity)
-   Semantic features
-   Expected‑item count for picture description
-   **Text embeddings** of `[question + answer]`

**Purpose:** captures *what* they say.

---

## **3.3 LLM‑Derived Clinical Scores**

LLM rates each answer (0–5 scales):

-   Coherence
-   Fluency
-   Semantic richness
-   Disorientation
-   Perseveration

Output is structured JSON.

**Purpose:** interpretable, clinically meaningful dimensions.

---

## **3.4 Task / Interaction Features**

-   Question type embedding
-   Response latency
-   Answer duration
-   Confusion markers (if detected)

---

## **3.5 Final Per‑Answer Vector**

```
answer_feature_i = concat(
  acoustic_stats_i,
  audio_embedding_i,
  linguistic_stats_i,
  text_embedding_i,
  LLM_scores_i,
  task_metadata_i
)
```

---

# **4️⃣ Session-Level Fusion over 5 Questions**

### **How it works**

-   Add positional/task embeddings
-   Pass the 5 vectors through a small transformer
-   Model learns which questions matter most
-   Output:
    -   `h_session` (session embedding)
    -   Attention weights (interpretability)

### **Why it matters**

Cognitive impairment is uneven across tasks.  
Transformer lets the model *focus on the most revealing answers*.

---

# **5️⃣ Longitudinal Modeling Across Sessions**

### **Input**

`h_session_t1 … h_session_tn`

### **How**

-   Sort by timestamp
-   Either:
    -   Transformer/GRU over time
    -   Or explicit trend features (slopes per metric)

### **Why it matters**

Changes over time matter more than one snapshot.  
Predictive accuracy increases massively for multi-session patients.

---

# **6️⃣ Multi-Task Prediction**

Uses:

-   `h_session`
-   Trend embeddings
-   Demographics

### **Outputs**

-   **Diagnosis probabilities**: HC / MCI / AD
-   **Severity regression**: predicted MMSE or MoCA
-   **NCRI (Neurolens Cognitive Risk Index)**: 0–100 consolidated risk score

### **Why multi-task?**

Joint classification + regression → smoother learning + better generalization.

---

# **7️⃣ Uncertainty, Quality Gating & Safety Checks**

### **Quality gating**

Reject or mark sessions with:

-   Too short audio
-   Low SNR
-   Low ASR confidence

### **Uncertainty estimation**

-   Probability entropy
-   Ensembling / MC-dropout

### **Bias checks**

Monitor performance by:

-   Age
-   Education
-   Language/dialect

### **Why this matters**

Prevents confident wrong predictions on low-quality or biased input.

---

# **8️⃣ Clinician-Facing Report Generation**

### **Inputs**

-   Diagnosis probabilities
-   Severity estimate
-   NCRI
-   Attention weights
-   Key acoustic & linguistic features
-   Longitudinal trends

### **Outputs**

#### **Structured Report**

-   Risk category
-   NCRI
-   Predicted MMSE
-   Diagnosis probabilities
-   Trend summary

#### **Narrative Explanation (LLM-constrained)**

Example:

> “This session shows high risk for cognitive impairment.  
> Most signal came from the verbal fluency and picture description tasks, showing long pauses and low information density.  
> Recommend full neuropsychological assessment.”

### **Purpose**

Turns AI math → something a clinician can trust and understand.

---

## **End of Pipeline**

Neurolens AI transforms 5 answers into a fully interpreted cognitive profile with risk, severity, and trajectory — backed by multimodal signals, LLM scoring, longitudinal modeling, and strong safety valves.

# 🧠 Neurolens AI – Cognitive Assessment Pipeline

Neurolens AI is a speech‑based, multimodal pipeline that turns **5 prompted answers** into an interpretable estimate of a patient’s **cognitive status, severity, and risk trajectory**.

---

## 🚀 At a Glance

-   Input: 5 short, structured Q&A audio responses from a patient
-   Output:
    -   HC / MCI / AD **risk probabilities**
    -   Estimated **severity** (e.g. MMSE / MoCA‑like score)
    -   A single **Neurolens Cognitive Risk Index (NCRI)**
    -   A **clinician‑friendly report** with explanations and trends

Neurolens is designed to be:

-   **Multimodal** – acoustic, linguistic, and LLM‑derived features
-   **Longitudinal** – tracks change over time, not just one visit
-   **Interpretable** – surfaces what questions and features drove the decision
-   **Safety‑aware** – uncertainty + quality gating to avoid overconfident BS

---

## 📦 Pipeline Overview

1.  **Session Ingestion & Metadata Capture**
2.  **Audio Preprocessing & ASR with Timestamps**
3.  **Multimodal Feature Extraction**
4.  **Session‑Level Fusion over 5 Questions**
5.  **Longitudinal Modeling Across Sessions**
6.  **Multi‑Task Prediction (Diagnosis, Severity, Risk Index)**
7.  **Uncertainty, Quality Gating & Safety Checks**
8.  **Clinician‑Facing Report Generation**

The sections below walk through each stage.

---

## 1️⃣ Session Ingestion & Metadata Capture

### Input

One “session” = the 5 question–answer pairs for a patient on a given day:

-   Question text + **question type**  
    (picture description, memory, orientation, fluency, executive, etc.)
-   Raw **answer audio** (per question)
-   (Eventually) **transcript** from ASR
-   **Patient context** (non‑identifiable where possible):
    -   Age, sex
    -   Education level
    -   Primary language / dialect
    -   High‑level diagnoses (HC / SMC / MCI / AD)
-   **Session metadata**:
    -   Device type, environment flags
    -   Timestamp (for longitudinal ordering)

### Output

A clean, structured JSON‑like object, e.g.:

```json
{
  "patient_id": "...",
  "session_id": "...",
  "timestamp": "...",
  "demographics": {...},
  "clinical_context": {...},
  "interactions": [
    {
      "q_id": "...",
      "question_text": "...",
      "question_type": "...",
      "audio_path": "...",
      "raw_transcript": null,
      "call_metadata": {...}
    }
  ]
}
```

### Why it matters

-   Defines **exactly what the model can see** (and not see).
-   Enforces a consistent schema → fewer bugs and less silent data leakage.
-   Allows education‑ and language‑aware interpretation of speech patterns.

---

## 2️⃣ Audio Preprocessing & ASR with Timestamps

### Core steps (per answer)

-   **Voice Activity Detection (VAD)**
    -   Trim leading/trailing silence, optionally compress long pauses.
-   **Diarization (if needed)**
    -   Separate patient vs clinician/agent; keep only patient channel.
-   **Noise reduction & normalization**
    -   Basic denoising + loudness normalization.
-   **ASR with word‑level timestamps**
    -   Transcript
    -   Word / segment timings
    -   ASR confidence scores

### Extra quality metrics

-   Total speech duration
-   Simple SNR proxy
-   Global ASR confidence / perplexity proxy

These feed into quality gating later.

### Why it matters

Everything else – lexical features, timing, LLM analysis – depends on:

-   Having the **right speaker**
-   Having a **usable transcript**
-   Knowing **when** each word was said

Garbage here = confident but wrong predictions downstream.

---

## 3️⃣ Multimodal Feature Extraction

Goal: turn each answer from “audio + text” into a **single feature vector**.

For each of the 5 answers we compute:

### 3.1 Acoustic Features

From the waveform:

-   Prosody & rhythm:
    -   Speech rate, articulation rate
    -   Pause counts, mean pause length, long‑pause count
    -   Ratio of speech time / total time
-   Voice quality:
    -   Mean pitch, pitch variability
    -   Simple jitter/shimmer‑style proxies
-   **SSL audio embeddings** (e.g. Whisper encoder / wav2vec2):
    -   Pooled vector representing “how this answer sounds”.

> Captures *how* they speak, independent of text.

---

### 3.2 Linguistic Features

From the ASR transcript (+ question text):

-   Lexical / syntactic:
    -   Type–token ratio (lexical richness)
    -   Avg sentence length, clause counts
    -   POS distribution / syntactic depth (if needed)
-   Semantic / discourse:
    -   Idea density (info per word)
    -   Repetition / tangents
    -   For picture description: fraction of expected key items mentioned
-   **Text embeddings**:
    -   Encode `[question_text] + [answer_transcript]` into a dense semantic vector.

> Captures *what* they say.

---

### 3.3 LLM‑Derived Clinical Scores

We **don’t** ask the LLM “does this person have dementia?”.  
Instead, we ask it to rate specific dimensions, e.g. 0–5 scales:

-   Coherence
-   Fluency
-   Semantic richness
-   Disorientation / confusion
-   Perseveration / repetition

We prompt the LLM to output **structured JSON** only.

> This creates interpretable, clinically meaningful features like  
> “coherence: 2/5, semantic richness: 1/5”.

---

### 3.4 Task / Interaction Features

From timing + question design:

-   Encoded **question type** (memory / fluency / etc.)
-   Response latency (end of question → start of speech)
-   Answer duration
-   Flags for confusion, restarts, asking for repetition (if detectable)

---

### 3.5 Final Per‑Answer Vector

All signals get fused into one vector:

```text
answer_feature_i = concat(
  acoustic_stats_i,
  audio_embedding_i,
  linguistic_stats_i,
  text_embedding_i,
  LLM_scores_i,
  task_metadata_i
)
```

This is the “compressed brain print” for that one answer.

---

## 4️⃣ Session‑Level Fusion over 5 Questions

Now we have 5 vectors: `answer_feature_1 .. answer_feature_5`.

### How we fuse them

-   Attach **task / positional embeddings** so the model knows which Q is which.
-   Feed the 5 vectors into a small **transformer / attention block**:
    -   Learns which questions are most informative per patient.
    -   Can pick up patterns like:
        -   “Memory is intact, but verbal fluency is badly impaired.”
        -   “Picture description is empty and disorganized.”

### Outputs

-   A single **session embedding** `h_session`
-   **Attention weights** over the 5 questions for interpretability

### Why it matters

Cognitive impairment is **not uniform** across tasks.  
Letting the model focus on the most revealing questions beats naive averaging.

---

## 5️⃣ Longitudinal Modeling Across Sessions

This is where Neurolens goes beyond one‑off benchmarks.

### Input

For each patient over time:

```text
h_session_t1, h_session_t2, ..., h_session_tn
```

### Options

-   **Temporal model** (GRU / transformer over sessions) to get a trajectory embedding, and/or
-   Explicit **trend features**:
    -   Slopes of key metrics (speech rate, idea density, NCRI, etc.)
    -   Δ per month or per visit

### Why it’s crucial

-   Some patients are **stable but borderline** → low risk.
-   Others look “okay” now but are **steadily declining** → high future risk.

Longitudinal info:

-   Sharpens severity estimates
-   Improves progression prediction (e.g. risk of converting MCI → AD)

---

## 6️⃣ Multi‑Task Prediction

Once we have:

-   `h_session`
-   Longitudinal / trend features
-   Demographic / clinical context

…we feed them into a multi‑head prediction module.

### Outputs

-   **Diagnosis probabilities**
    -   e.g. P(HC), P(SMC), P(MCI), P(AD)
-   **Severity regression**
    -   Predicted MMSE / MoCA / CDR‑SB‑like score (with uncertainty)
-   **Neurolens Cognitive Risk Index (NCRI)**
    -   A 0–100 risk index combining:
        -   Current severity
        -   Longitudinal trend
        -   Model uncertainty

### Why multi‑task?

Jointly training classification + regression:

-   Encourages the model to learn a **continuous spectrum** of impairment
-   Usually improves generalization, especially around borderline cases

---

## 7️⃣ Uncertainty, Quality Gating & Safety

This stage exists so we **don’t ruin lives with overconfident guesses**.

### 7.1 Quality Gating

Uses earlier metrics:

-   Minimum speech duration
-   SNR bounds
-   ASR confidence thresholds

If quality is too low:

-   Suppress hard labels, or
-   Mark the session as **“low‑quality – please retest”**.

---

### 7.2 Uncertainty Estimation

Even with good audio, the model can be unsure. We use:

-   Probability entropy / confidence
-   Calibrated logits (e.g. temperature scaling)
-   Optional ensembles or MC‑dropout

Logic examples:

-   **High risk + high uncertainty** → escalate for full neuropsych eval.
-   **Low risk + high uncertainty** → suggest repeat testing, not “you’re fine”.

---

### 7.3 Bias & Subgroup Monitoring (Offline)

Continually monitor performance across:

-   Age groups
-   Education levels
-   Language / dialect groups

If a subgroup underperforms:

-   Adjust thresholds
-   Prioritize data collection for that subgroup
-   Narrow or qualify deployment claims

---

## 8️⃣ Clinician‑Facing Report Generation

Last step: turn all the math into something a clinician can actually use.

### Inputs to the report generator

-   Diagnosis probabilities
-   Severity estimate (+ uncertainty)
-   NCRI
-   Attention weights over questions
-   Key acoustic + linguistic stats
-   Longitudinal change (if multiple sessions exist)

### 8.1 Structured Report

Example contents:

-   **Risk category:** Low / Moderate / High
-   **NCRI:** e.g. `73 / 100`
-   **Diagnosis probabilities:**
    -   `P(HC)=0.12, P(MCI)=0.61, P(AD)=0.27`
-   **Estimated severity:**
    -   e.g. “Estimated MMSE: 23 ± 2”
-   **Trajectory:**
    -   “Speech rate decreased ~18% over 12 months”
    -   “Idea density dropped ~12% since last visit”

---

### 8.2 Narrative Explanation (LLM‑Powered, Read‑Only)

A tightly‑prompted LLM turns the above into a short narrative, e.g.:

> “This session is classified as **high risk** for early dementia.  
> Most signal comes from the verbal fluency and picture description tasks,  
> where the patient shows frequent long pauses and low information content  
> relative to their previous sessions.  
> Recommend: full neuropsychological assessment and follow‑up within 3 months.”

Important:

-   The LLM **does not change** the predictions.
-   It only **explains** them in clinical language.

---

## ✅ Summary

Neurolens AI takes 5 structured Q&A responses and produces:

-   A **risk estimate**,
-   A **severity estimate**,
-   A **trajectory‑aware risk index (NCRI)**, and
-   An **interpretable report** grounded in acoustic, linguistic, LLM‑derived, and longitudinal features.

All while enforcing **quality checks**, **uncertainty awareness**, and **transparency** so that clinicians can trust – and audit – what the system is doing.