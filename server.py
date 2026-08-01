"""flask server for the visable page"""

import io
import os
import re
import sys

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image
import torch
from transformers import BlipConfig, BlipForConditionalGeneration, BlipProcessor

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

BASE_BLIP_MODEL = "Salesforce/blip-image-captioning-base"
HUB_MODEL = os.environ.get("VISABLE_MODEL", "hkondle/VisAble-Diagram-Captioner")
LOCAL_MODEL_DIR = os.path.join(PROJECT_ROOT, "vision", "diagram_blip_model")

# penalties make it hallucinate
MAX_LENGTH = int(os.environ.get("VISABLE_MAX_LENGTH", "256"))
NUM_BEAMS = int(os.environ.get("VISABLE_NUM_BEAMS", "4"))
MIN_LENGTH = int(os.environ.get("VISABLE_MIN_LENGTH", "0"))
LENGTH_PENALTY = float(os.environ.get("VISABLE_LENGTH_PENALTY", "1.0"))
REPETITION_PENALTY = float(os.environ.get("VISABLE_REPETITION_PENALTY", "1.0"))
NO_REPEAT_NGRAM = int(os.environ.get("VISABLE_NO_REPEAT_NGRAM", "0"))

# keep one sentence minimum
MIN_CLIPPED_CHARS = 80

app = Flask(__name__, static_folder=None)
CORS(app)

_processor = None
_model = None
_device = None
_model_source = None


def tidy(text):
    """fix tokenizer spacing"""
    text = re.sub(r"\s+", " ", text).strip()

    # contractions
    text = re.sub(r"\s*'\s*(?=(?:s|t|d|m|ll|re|ve)\b)", "'", text)

    # plural possessives
    text = re.sub(r"\s+'", "'", text)

    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\s+&\s+", " and ", text)

    if text:
        text = text[0].upper() + text[1:]

    return text


def clip_to_last_sentence(text):
    """end on a period"""
    if text.endswith((".", "!", "?")):
        return text

    ends = list(re.finditer(r"[.!?](?=\s|$)", text))

    if ends:
        trimmed = text[: ends[-1].end()]
        if len(trimmed) >= MIN_CLIPPED_CHARS:
            return trimmed

    return text.rstrip(" ,;:-—") + "."


def get_device():
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def resolve_model_source():
    """prefer a local folder"""
    if os.path.isdir(LOCAL_MODEL_DIR):
        return LOCAL_MODEL_DIR

    return HUB_MODEL


def untied_config(source):
    """untie head from embeddings"""
    config = BlipConfig.from_pretrained(source)
    config.tie_word_embeddings = False
    config.text_config.tie_word_embeddings = False

    return config


def load_blip():
    global _processor, _model, _model_source

    if _model is not None:
        return _processor, _model

    source = resolve_model_source()
    print(f"loading model from {source}", flush=True)

    try:
        _processor = BlipProcessor.from_pretrained(source)
        _model = BlipForConditionalGeneration.from_pretrained(
            source, config=untied_config(source)
        )
        _model_source = source
    except Exception as error:
        print(f"could not load {source} {error}", flush=True)
        print("falling back to base model", flush=True)
        _processor = BlipProcessor.from_pretrained(BASE_BLIP_MODEL)
        _model = BlipForConditionalGeneration.from_pretrained(BASE_BLIP_MODEL)
        _model_source = BASE_BLIP_MODEL

    _model.to(get_device())
    _model.eval()

    print(f"model ready on {get_device()}", flush=True)

    return _processor, _model


def describe_image(image):
    processor, model = load_blip()
    device = get_device()

    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    options = {"max_length": MAX_LENGTH, "num_beams": NUM_BEAMS}

    if MIN_LENGTH > 0:
        options["min_length"] = MIN_LENGTH

    if LENGTH_PENALTY != 1.0:
        options["length_penalty"] = LENGTH_PENALTY

    if REPETITION_PENALTY != 1.0:
        options["repetition_penalty"] = REPETITION_PENALTY

    if NO_REPEAT_NGRAM > 0:
        options["no_repeat_ngram_size"] = NO_REPEAT_NGRAM

    with torch.no_grad():
        output_ids = model.generate(**inputs, **options)

    caption = clip_to_last_sentence(tidy(processor.decode(output_ids[0], skip_special_tokens=True)))

    if _model_source == BASE_BLIP_MODEL:
        # base model needs rewriting
        from data.modifying_data import generate_part_to_whole

        caption = tidy(generate_part_to_whole(caption))

    return caption


def audio_enabled():
    from vision.text_to_audio import audio_available

    return audio_available()


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model": resolve_model_source(),
            "loaded": _model is not None,
            "device": str(get_device()),
            "audio": audio_enabled(),
        }
    )


@app.route("/api/describe", methods=["POST"])
def describe():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    uploaded = request.files["image"]

    if not uploaded.filename:
        return jsonify({"error": "Empty filename"}), 400

    try:
        image = Image.open(io.BytesIO(uploaded.read()))
    except Exception:
        return jsonify({"error": "That file could not be read as an image"}), 400

    try:
        caption = describe_image(image)
    except Exception as error:
        return jsonify({"error": f"Description failed: {error}"}), 500

    return jsonify({"caption": caption, "model": _model_source})


@app.route("/api/speak", methods=["POST"])
def speak():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    if not audio_enabled():
        return jsonify({"error": "OPENAI_API_KEY is not set on the server."}), 503

    from vision.text_to_audio import text_to_audio

    output_path = os.path.join(PROJECT_ROOT, "output.mp3")

    try:
        text_to_audio(text, output_path=output_path)
    except Exception as error:
        return jsonify({"error": f"Audio generation failed: {error}"}), 500

    return send_file(output_path, mimetype="audio/mpeg")


if __name__ == "__main__":
    print("serving on http://localhost:5001", flush=True)
    app.run(host="0.0.0.0", port=5001, debug=False)
