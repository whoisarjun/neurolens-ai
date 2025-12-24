# Extraction of LLM scores

import json
import re
from pathlib import Path

import numpy as np
from ollama import chat
from ollama import ChatResponse

from utils import cache

CACHE_DIR = Path('cache/semantics')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL = 'qwen2.5:7b'

DEFAULT_SEMANTIC_SCORE = 1.0

def _ask(prompt: str, model=MODEL):
    response: ChatResponse = chat(model=MODEL, messages=[
      {
        'role': 'user',
        'content': prompt,
      },
    ], options={'temperature': 0, 'top_p': 1, 'repeat_penalty': 1.0})
    return response['message']['content']

def _prompt(question: str, transcript: str, features, model=MODEL):
    features_json = json.dumps(features, ensure_ascii=False, indent=2)
    prompt = f"""
You are an expert clinical language evaluator specializing in early-stage dementia.

You will:
1. Read the interview QUESTION.
2. Read the patient's TRANSCRIPT.
3. Read the list of FEATURES and their 0–4 scoring rubrics.
4. For EACH FEATURE, think step-by-step under a section called REASONING.
5. Then, under OUTPUT, return ONLY a single JSON object matching the schema.

RULES:
- NO markdown.
- NO backticks.
- Do NOT wrap anything in ```json or any other fences.
- After the JSON, do NOT add any extra text.

Format:

REASONING:
[Write your internal reasoning for each feature here in plain text.]

OUTPUT:
{{
  "scores": [
    {{"feature": "FEATURE_NAME", "score": 0-4}},
    ...
  ]
}}

QUESTION:
{question}

TRANSCRIPT:
{transcript}

FEATURES:
{features_json}
"""
    return _ask(prompt, model)

def _parse_scores(raw: str) -> list[float]:
    if "OUTPUT:" in raw:
        raw = raw.split("OUTPUT:", 1)[1].strip()

    raw = re.sub(r"^```[a-zA-Z0-9]*", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    if not raw:
        print("LLM returned no JSON.")
        raise LLMParseError("Empty JSON payload from LLM")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise LLMParseError(str(e))

    return [
        float(s) if isinstance((s := item.get('score')), (int, float)) and 0 <= s <= 4
        else DEFAULT_SEMANTIC_SCORE
        for item in data['scores']
    ]

class LLMParseError(Exception):
    # if the llm reponse cannot be parsed after 3 tries
    pass

# ========== FEATURE EXTRACTION ========== #

feature_list = [
    {
        'name': 'Semantic Memory Degradation',
        'description': 'Measures factual correctness, misremembering, fabricated elements, and confabulation of details.',
        'scale': {
            0: 'Severe factual errors, fabricated memories, impossible/confused details',
            1: 'Frequent factual errors or misremembering concrete details',
            2: 'Occasional errors or contradictions',
            3: 'Mostly accurate with rare slips',
            4: 'Fully accurate, grounded in real-world facts'
        }
    },
    {
        'name': 'Narrative Structure Disintegration',
        'description': 'Evaluates temporal order, causal flow, and narrative organization.',
        'scale': {
            0: 'No temporal order, random fragments, no causal links',
            1: 'Frequent jumps, broken timeline, major missing steps',
            2: 'Mixed: some order but several breakdowns',
            3: 'Mostly logical progression with minor tangents',
            4: 'Clear temporal and causal structure'
        }
    },
    {
        'name': 'Pragmatic Appropriateness',
        'description': 'Measures whether the patient answers the intended question with appropriate detail and tone.',
        'scale': {
            0: 'Totally mismatched response; irrelevant or inappropriate content',
            1: 'Frequent mismatches; over/under explanations',
            2: 'Mixed pragmatics; some mismatches',
            3: 'Mostly appropriate; minor issues',
            4: 'Fully appropriate and aligned with the question’s intent'
        }
    },
    {
        'name': 'Topic Maintenance',
        'description': 'Ability to stay on topic without drifting into unrelated content.',
        'scale': {
            0: 'Constant, rapid derailment',
            1: 'Frequent drift to unrelated topics',
            2: 'Some drift; partially maintained topic',
            3: 'Mostly on topic; rare drift',
            4: 'Completely sustained topic focus'
        }
    },
    {
        'name': 'Perseveration Types',
        'description': 'Repetitions beyond normal speech including stuck loops, intrusive details, or returning to ideas minutes later.',
        'scale': {
            0: 'Severe stuck loops, repeated ideas despite prompts',
            1: 'Frequent intrusive repetition',
            2: 'Occasional repetition',
            3: 'Mild redundancy',
            4: 'No noticeable perseveration'
        }
    },
    {
        'name': 'Disorientation Types',
        'description': 'Temporal, spatial, or personal confusion detectable through inconsistencies.',
        'scale': {
            0: 'Severe temporal/spatial/personal confusion',
            1: 'Frequent incorrect references',
            2: 'Occasional confusion or mixing details',
            3: 'Mostly oriented',
            4: 'Fully oriented'
        }
    },
    {
        'name': 'Executive Dysfunction Patterns',
        'description': 'Ability to follow tasks, plan, structure answers, and avoid irrelevant content.',
        'scale': {
            0: 'Does not follow task; chaotic, aimless answer',
            1: 'Major task-switching failures; irrelevant answers',
            2: 'Some executive issues',
            3: 'Minor slips but overall intact',
            4: 'Fully goal-directed and task-oriented'
        }
    },
    {
        'name': 'Abstract Reasoning',
        'description': 'Ability to generalize, draw analogies, and interpret beyond literal meaning.',
        'scale': {
            0: 'Cannot reason abstractly; fully literal; major failures',
            1: 'Frequent failures to generalize/abstract',
            2: 'Mixed abstraction with some literal interpretations',
            3: 'Mostly good abstraction; minor issues',
            4: 'Fully capable of abstraction and generalization'
        }
    },
    {
        'name': 'Semantic Clustering vs Fragmentation',
        'description': 'Degree of semantic cohesion between sentences.',
        'scale': {
            0: 'Fragmented, isolated sentences, no cohesion',
            1: 'Frequent breaks in semantic glue',
            2: 'Equal mix of cohesion and fragmentation',
            3: 'Mostly cohesive with minor breaks',
            4: 'Strong semantic clustering throughout'
        }
    },
    {
        'name': 'Emotional Appropriateness',
        'description': 'Whether emotional tone matches the content and context.',
        'scale': {
            0: 'Clearly inappropriate affect (flat, paranoid, manic, etc.)',
            1: 'Frequent mismatches in emotional tone',
            2: 'Occasional odd affect',
            3: 'Mostly appropriate',
            4: 'Fully appropriate emotional expression'
        }
    },
    {
        'name': 'Novel Information Content',
        'description': 'Amount of meaningful new information versus repetition.',
        'scale': {
            0: 'Almost no new information; repetitive or empty',
            1: 'Very low information density',
            2: 'Moderate idea density',
            3: 'Good information density; several meaningful ideas',
            4: 'High density; rich, detailed, informative'
        }
    },
    {
        'name': 'Ambiguity & Vagueness',
        'description': 'Overuse of vague references, circular speech, and empty language.',
        'scale': {
            0: 'Extreme vagueness; heavily ambiguous',
            1: 'Frequent vague references',
            2: 'Occasional vagueness',
            3: 'Mostly specific',
            4: 'Clear, precise, unambiguous'
        }
    },
    {
        'name': 'Instruction Following',
        'description': 'Accuracy in doing the task the question requires.',
        'scale': {
            0: 'Does not follow the question at all',
            1: 'Frequently misinterprets or answers unrelated parts',
            2: 'Partially follows instructions',
            3: 'Mostly follows with minor deviations',
            4: 'Fully follows instructions'
        }
    },
    {
        'name': 'Logical Self-Consistency',
        'description': 'Internal contradictions within the answer.',
        'scale': {
            0: 'Major contradictions within the response',
            1: 'Frequent inconsistencies or switching facts',
            2: 'Some inconsistencies',
            3: 'Mostly consistent',
            4: 'Fully self-consistent'
        }
    },
    {
        'name': 'Confabulation',
        'description': 'Invented events or plausible-but-false details stated confidently.',
        'scale': {
            0: 'Major, repeated invented details; fabricated stories',
            1: 'Frequent plausible-sounding but untrue content',
            2: 'Occasional embellishment or invented info',
            3: 'Rare or minor exaggerations',
            4: 'No signs of confabulation'
        }
    },
    {
        'name': 'Clinical Impression',
        'description': 'LLM’s clinical overall severity rating based on full discourse.',
        'scale': {
            0: 'Strong indication of severe cognitive impairment',
            1: 'Moderate–severe impairment',
            2: 'Mild cognitive impairment',
            3: 'Borderline normal with slight weaknesses',
            4: 'Clearly cognitively intact'
        }
    },
    {
        'name': 'Error Type Classification',
        'description': 'Global severity of linguistic, semantic, phonological, retrieval, and executive errors.',
        'scale': {
            0: 'Severe errors across multiple types',
            1: 'Frequent multi-category errors',
            2: 'Moderate errors',
            3: 'Minor errors',
            4: 'No notable errors'
        }
    },
    {
        'name': 'Compensation Strategies',
        'description': 'Avoidance strategies, circumlocution, meta-comments about memory, and self-correction behaviors.',
        'scale': {
            0: 'Severe reliance on compensation strategies',
            1: 'Frequent compensation cues',
            2: 'Occasional compensation',
            3: 'Mild compensation',
            4: 'No noticeable compensatory behavior'
        }
    }
]

def _ask_features(question: str, transcript: dict, features: list[dict], model=MODEL):
    sections = [
        features[0:4],
        features[4:8],
        features[8:12],
        features[12:15],
        features[15:18]
    ]

    def process_section(s):
        last_error = None
        for attempt in range(3):
            try:
                raw = _prompt(question, transcript.get('text', ''), s, model)
                return _parse_scores(raw)
            except LLMParseError as e:
                # smth wrong with the transcript → no point retrying 3 times
                last_error = e
                break
            except Exception as e:
                # transient LLM/backend error → retry a few times
                last_error = e
                continue

        if isinstance(last_error, LLMParseError):
            # propagate up so caller can react (re-ASR, etc.)
            raise last_error

        # if it's some other persistent error, you can still fall back to 1/4
        return default_semantic_features()

    results = []
    for s in sections:
        section_scores = process_section(s)
        results.append(section_scores)

    return (lambda sc: sc[:len(feature_list)] + [DEFAULT_SEMANTIC_SCORE] * max(0, len(feature_list) - len(sc)))(
        [score for section_scores in results for score in section_scores])

# ========== COMBINE EVERYTHING ========== #g

def extract(question: str, transcript: dict, filename: Path, use_cache=False, model=MODEL, save=True):
    scores = None
    cache_file = cache.key(filename, CACHE_DIR)
    if use_cache:
        scores = cache.load(cache_file)
        # print(f"[SEM] Loaded from cache: {scores is not None}, file: {cache_file.name}")  # debug
    if scores is None:
        # this can now raise LLMParseError
        scores = np.array(_ask_features(question, transcript, feature_list, model), dtype=np.float32)
        if save:
            cache.save(cache_file, scores)

    return scores

def default_semantic_features() -> np.array:
    return np.full(len(feature_list), DEFAULT_SEMANTIC_SCORE, dtype=np.float32)

