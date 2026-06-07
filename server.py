from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from flask import Flask, jsonify, request
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from features import acoustics, linguistics, semantics
from ml import model
from processing import cleanup, transcriber
from utils.language import normalize_language


app = Flask(__name__)

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok'
    }), 200

REG_WEIGHTS_PATH = Path("models/model_weights_reg.pth")
CLS_WEIGHTS_PATH = Path("models/model_weights_cls.pth")
SCALER_PATH = Path("models/model_scaler.pkl")

EXPECTED_FILE_COUNT = 5
EXPECTED_FEATURE_DIM = 52 + 29 + 18 + 1024
NAMED_FEATURE_COUNT = 52 + 29 + 18
FEATURE_NAMES = [
    'mean_f0',
    'std_f0',
    'min_f0',
    'max_f0',
    'f0_iqr',
    'mean_energy',
    'std_energy',
    'dynamic_range',
    'syllables_per_sec',
    'words_per_sec',
    'pause_count_over_threshold',
    'total_pause_duration',
    'pause_ratio',
    'mfcc1_mean',
    'mfcc1_std',
    'mfcc2_mean',
    'mfcc2_std',
    'mfcc3_mean',
    'mfcc3_std',
    'mfcc4_mean',
    'mfcc4_std',
    'mfcc5_mean',
    'mfcc5_std',
    'mfcc6_mean',
    'mfcc6_std',
    'mfcc7_mean',
    'mfcc7_std',
    'mfcc8_mean',
    'mfcc8_std',
    'mfcc9_mean',
    'mfcc9_std',
    'mfcc10_mean',
    'mfcc10_std',
    'mfcc11_mean',
    'mfcc11_std',
    'mfcc12_mean',
    'mfcc12_std',
    'mfcc13_mean',
    'mfcc13_std',
    'spectral_centroid_mean',
    'spectral_centroid_std',
    'spectral_bandwidth_mean',
    'spectral_bandwidth_std',
    'spectral_flux_mean',
    'spectral_flux_std',
    'spectral_slope',
    'jitter',
    'shimmer',
    'harmonics_to_noise_ratio',
    'cepstral_peak_prominence',
    'zero_crossing_rate_mean',
    'zero_crossing_rate_std',
    'total_tokens',
    'unique_tokens',
    'type_token_ratio',
    'mean_words_per_utterance',
    'max_utterance_length',
    'number_of_sentences',
    'content_word_ratio',
    'function_word_ratio',
    'rare_word_ratio',
    'filler_word_count',
    'repetition_score',
    'bigram_repetition_ratio',
    'self_correction_count',
    'mean_local_coherence',
    'coherence_variance',
    'mean_dependency_distance',
    'clause_density',
    'mean_parse_tree_height',
    'pronoun_ratio',
    'verb_to_noun_ratio',
    'auxiliary_verb_ratio',
    'idea_density',
    'mean_concreteness',
    'abstract_ratio',
    'flesch_kincaid_grade',
    'mean_syllables',
    'long_word_ratio',
    'global_coherence_drift',
    'topic_recurrence',
    'semantic_memory_degradation',
    'narrative_structure_disintegration',
    'pragmatic_appropriateness',
    'topic_maintenance',
    'perseveration_types',
    'disorientation_types',
    'executive_dysfunction_patterns',
    'abstract_reasoning',
    'semantic_clustering_vs_fragmentation',
    'emotional_appropriateness',
    'novel_information_content',
    'ambiguity_and_vagueness',
    'instruction_following',
    'logical_self_consistency',
    'confabulation',
    'clinical_impression',
    'error_type_classification',
    'compensation_strategies',
]

if len(FEATURE_NAMES) != NAMED_FEATURE_COUNT:
    raise RuntimeError(
        f'FEATURE_NAMES has length {len(FEATURE_NAMES)}; expected {NAMED_FEATURE_COUNT}'
    )


def _format_feature_response(average_feature: np.ndarray) -> dict[str, float | list[float]]:
    if average_feature.shape[0] != EXPECTED_FEATURE_DIM:
        raise RuntimeError(
            f"unexpected averaged feature length {average_feature.shape[0]}; expected {EXPECTED_FEATURE_DIM}"
        )

    named_features = {
        feature_name: float(value)
        for feature_name, value in zip(
            FEATURE_NAMES,
            average_feature[:NAMED_FEATURE_COUNT],
            strict=True,
        )
    }
    audio_embeddings = average_feature[NAMED_FEATURE_COUNT:].astype(np.float32).tolist()

    return {
        **named_features,
        "audio_embeddings": audio_embeddings,
    }

SUPPORTED_LANGUAGES = {"en", "zh"}

INFERENCE_LOCK = Lock()

backbone = None
regressor = None
classifier = None


def _json_error(message: str, status_code: int):
    response = jsonify({"error": message})
    response.status_code = status_code
    return response


def _validate_language(raw_language: str | None) -> str:
    language = normalize_language(raw_language, default="__invalid__")
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError("language must be one of: en, zh")
    return language


def _validate_request_payload():
    files = request.files.getlist("files")
    questions = request.form.getlist("questions")
    language = _validate_language(request.form.get("language"))

    if len(files) != EXPECTED_FILE_COUNT:
        raise ValueError(f"exactly {EXPECTED_FILE_COUNT} audio files are required")

    if len(questions) != EXPECTED_FILE_COUNT:
        raise ValueError(f"exactly {EXPECTED_FILE_COUNT} questions are required")

    empty_files = [idx for idx, file in enumerate(files, start=1) if not isinstance(file, FileStorage) or not file.filename]
    if empty_files:
        raise ValueError("all uploaded files must have filenames")

    return files, questions, language


def _safe_upload_name(upload: FileStorage, index: int) -> str:
    original_name = secure_filename(upload.filename or "")
    if original_name:
        return f"{index:02d}_{original_name}"
    return f"{index:02d}.wav"


def _build_feature_vector(upload: FileStorage, question: str, language: str, work_dir: Path, index: int) -> np.ndarray:
    input_path = work_dir / _safe_upload_name(upload, index)
    cleaned_path = work_dir / f"{input_path.stem}_cleaned.wav"

    upload.save(input_path)

    cleanup.normalize(input_path, cleaned_path, verbose=False)
    cleanup.denoise(cleaned_path, verbose=False)

    transcript = transcriber.asr(cleaned_path, use_cache=False, language=language)
    acoustic_features = acoustics.extract(cleaned_path, transcript, use_cache=False, verbose=False)
    linguistic_features = linguistics.extract(
        cleaned_path,
        transcript,
        use_cache=False,
        verbose=False,
        language=language,
    )

    try:
        semantic_features = semantics.extract(
            question,
            transcript,
            cleaned_path,
            use_cache=False,
            save=False,
        )
    except semantics.LLMParseError:
        semantic_features = semantics.default_semantic_features(language=language)

    embedding_tensor = transcriber.embeddings(
        [cleaned_path],
        use_cache=False,
        batch_size=1,
        tqdm_desc="Generating embeddings",
    )[cleaned_path]
    embedding_features = embedding_tensor.numpy().ravel().astype(np.float32)

    feature_vector = np.concatenate(
        [acoustic_features, linguistic_features, semantic_features, embedding_features]
    ).astype(np.float32)

    if feature_vector.shape[0] != EXPECTED_FEATURE_DIM:
        raise RuntimeError(
            f"unexpected feature vector length {feature_vector.shape[0]}; expected {EXPECTED_FEATURE_DIM}"
        )

    return feature_vector


def _load_inference_models():
    global backbone, regressor, classifier

    if backbone is not None and regressor is not None and classifier is not None:
        return

    missing = [str(path) for path in (SCALER_PATH, REG_WEIGHTS_PATH, CLS_WEIGHTS_PATH) if not path.exists()]
    if missing:
        raise RuntimeError(f"missing inference artifacts: {', '.join(missing)}")

    model.load_scaler(SCALER_PATH)

    backbone = model.new_backbone()
    regressor, classifier, _, _, _, _ = model.new_multitask(backbone)
    model.load(REG_WEIGHTS_PATH, regressor)
    model.load(CLS_WEIGHTS_PATH, classifier)

    regressor.eval()
    classifier.eval()


def _predict_from_average_feature(average_feature: np.ndarray):
    scaled_feature = model.transform_features([average_feature]).astype(np.float32)
    feature_tensor = torch.tensor(scaled_feature, dtype=torch.float32, device=model.device)

    with torch.no_grad():
        mmse_pred = float(regressor(feature_tensor).cpu().item())
        logits = classifier(feature_tensor)
        cog_idx = int(torch.argmax(logits, dim=1).cpu().item())

    return mmse_pred, model.cog_statuses[cog_idx]


@app.route("/predict", methods=["POST"])
def predict():
    try:
        files, questions, language = _validate_request_payload()
    except ValueError as exc:
        return _json_error(str(exc), 400)

    try:
        _load_inference_models()
    except RuntimeError as exc:
        return _json_error(str(exc), 500)

    temp_dir = Path(tempfile.mkdtemp(prefix="neurolens_inference_"))

    try:
        with INFERENCE_LOCK:
            feature_vectors = [
                _build_feature_vector(upload, question, language, temp_dir, index)
                for index, (upload, question) in enumerate(zip(files, questions, strict=True), start=1)
            ]
            average_feature = np.mean(np.stack(feature_vectors, axis=0), axis=0).astype(np.float32)
            formatted_features = _format_feature_response(average_feature)
            mmse_pred, cog_status = _predict_from_average_feature(average_feature)
    except Exception as exc:
        return _json_error(f"inference failed: {exc}", 500)
    finally:
        # shutil.rmtree(temp_dir, ignore_errors=True)
        pass

    return jsonify(
        {
            "features": formatted_features,
            "mmse": round(mmse_pred, 4),
            "cog_status": cog_status,
        }
    )


if __name__ == "__main__":
    _load_inference_models()
    app.run(host="0.0.0.0", port=6700)
