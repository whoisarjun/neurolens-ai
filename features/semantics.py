# Extraction of LLM scores

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from ollama import ChatResponse, chat

from utils import cache
from utils.language import normalize_language

CACHE_DIR = Path('cache/semantics')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_LIST_DIR = Path(__file__).with_name('semantic_feature_lists')
MODEL = 'ministral-3:8b'
DEFAULT_SEMANTIC_SCORE = 1.0

def _ask(prompt: str, model=MODEL):
    response: ChatResponse = chat(
        model=model,
        messages=[{
            'role': 'user',
            'content': prompt,
        }],
        options={'temperature': 0, 'top_p': 1, 'repeat_penalty': 1.0}
    )
    return response['message']['content']

@lru_cache(maxsize=2)
def _load_feature_list(language: str) -> list[dict]:
    lang = normalize_language(language)
    path = FEATURE_LIST_DIR / f'semantic_feature_list_{lang}.json'
    if not path.exists() and lang != 'en':
        path = FEATURE_LIST_DIR / 'semantic_feature_list_en.json'

    payload = json.loads(path.read_text(encoding='utf-8'))
    features = payload.get('features', [])
    if not isinstance(features, list) or not features:
        raise ValueError(f'Invalid semantic feature list: {path}')
    return features

def _prompt(question: str, transcript: str, features: list[dict], language: str, model=MODEL):
    features_json = json.dumps(features, ensure_ascii=False, indent=2)

    if language == 'zh':
        prompt = f"""
你是一位专门评估早期失智语言表现的临床语言专家。

你的任务：
1. 阅读访谈问题 QUESTION。
2. 阅读患者回答 TRANSCRIPT。
3. 阅读待评分的 FEATURES 及其 0–4 评分标准。
4. 对每个特征进行逐步分析，放在 REASONING 部分。
5. 最后在 OUTPUT 部分只返回一个符合 schema 的 JSON 对象。

规则：
- 不要使用 markdown。
- 不要使用反引号。
- 不要把内容包在 ```json 或任何代码块里。
- JSON 后不要再添加任何额外文字。

格式：

REASONING:
[在这里写出每个特征的分析]

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
    if 'OUTPUT:' in raw:
        raw = raw.split('OUTPUT:', 1)[1].strip()

    raw = re.sub(r'^```[a-zA-Z0-9]*', '', raw).strip()
    raw = re.sub(r'```$', '', raw).strip()

    if not raw:
        raise LLMParseError('Empty JSON payload from LLM')

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMParseError(str(exc))

    return [
        float(score) if isinstance(score, (int, float)) and 0 <= score <= 4
        else DEFAULT_SEMANTIC_SCORE
        for item in data['scores']
        for score in [item.get('score')]
    ]

class LLMParseError(Exception):
    pass

def _ask_features(question: str, transcript: dict, features: list[dict], language: str, model=MODEL):
    sections = [
        features[0:4],
        features[4:8],
        features[8:12],
        features[12:15],
        features[15:18],
    ]

    def process_section(section: list[dict]):
        last_error = None
        for _ in range(3):
            try:
                raw = _prompt(question, transcript.get('text', ''), section, language=language, model=model)
                return _parse_scores(raw)
            except LLMParseError as exc:
                last_error = exc
                break
            except Exception as exc:
                last_error = exc
                print(exc)
                continue

        if isinstance(last_error, LLMParseError):
            raise last_error

        return [DEFAULT_SEMANTIC_SCORE] * len(section)

    results = []
    for section in sections:
        results.extend(process_section(section))

    return results[:len(features)] + [DEFAULT_SEMANTIC_SCORE] * max(0, len(features) - len(results))

def extract(question: str, transcript: dict, filename: Path, use_cache=False, model=MODEL, save=True):
    language = normalize_language(transcript.get('language'))
    feature_list = _load_feature_list(language)
    cache_variant = None if language == 'en' else f'sem:{language}'

    scores = None
    cache_file = cache.key(filename, CACHE_DIR, variant=cache_variant)
    if use_cache:
        scores = cache.load(cache_file)
    if scores is None:
        scores = np.array(_ask_features(question, transcript, feature_list, language=language, model=model), dtype=np.float32)
        if save:
            cache.save(cache_file, scores)

    return scores

def default_semantic_features(language: str = 'en') -> np.array:
    return np.full(len(_load_feature_list(language)), DEFAULT_SEMANTIC_SCORE, dtype=np.float32)
