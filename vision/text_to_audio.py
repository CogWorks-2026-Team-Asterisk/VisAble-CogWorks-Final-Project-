"""read a description aloud"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.path.join(PROJECT_ROOT, ".env")

# only this model takes instructions
MODEL = os.environ.get("VISABLE_TTS_MODEL", "gpt-4o-mini-tts")
VOICE = os.environ.get("VISABLE_TTS_VOICE", "marin")
SPEED = float(os.environ.get("VISABLE_TTS_SPEED", "0.95"))

# for older keys
FALLBACK_MODEL = "tts-1"
FALLBACK_VOICE = "alloy"

INSTRUCTIONS = (
    "You are reading a description of a diagram aloud to a blind or low vision "
    "student who cannot see the image. Speak warmly and unhurriedly in a clear "
    "teaching voice. Pause at the end of every sentence, and pause a little "
    "longer whenever a new part of the diagram is introduced, so the listener "
    "has time to build a mental picture. Give the names of the parts a gentle "
    "emphasis. Never sound rushed, clipped or robotic."
)

MAX_INPUT_CHARS = 4096

_client = None


def load_env_file():
    """read keys out of .env"""
    if not os.path.isfile(ENV_FILE):
        return

    with open(ENV_FILE) as handle:
        for line in handle:
            line = line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def get_api_key():
    load_env_file()
    return os.environ.get("OPENAI_API_KEY")


def audio_available():
    return bool(get_api_key())


def get_openai_client():
    global _client

    if _client is None:
        from openai import OpenAI

        api_key = get_api_key()
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is required to generate audio")

        _client = OpenAI(api_key=api_key)

    return _client


def text_to_audio(text, output_path="output.mp3"):
    """speak text into an mp3"""
    client = get_openai_client()
    text = text.strip()[:MAX_INPUT_CHARS]

    try:
        with client.audio.speech.with_streaming_response.create(
            model=MODEL,
            voice=VOICE,
            input=text,
            instructions=INSTRUCTIONS,
            speed=SPEED,
            response_format="mp3",
        ) as response:
            response.stream_to_file(output_path)

        return output_path
    except Exception as error:
        print(f"{MODEL} unavailable {error}", flush=True)
        print("falling back to tts-1", flush=True)

    with client.audio.speech.with_streaming_response.create(
        model=FALLBACK_MODEL,
        voice=FALLBACK_VOICE,
        input=text,
        response_format="mp3",
    ) as response:
        response.stream_to_file(output_path)

    return output_path
