import whisper

MODEL_NAME = "base"
_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = whisper.load_model(MODEL_NAME)
    return _MODEL


def transcribe_audio(file_path, progress_callback=None):
    """Transcribe audio and optionally report progress via `progress_callback`.

    `progress_callback` is a callable that accepts a single float between 0.0 and 1.0.
    This function will call the callback at a few logical steps so the caller can
    update job progress in the UI.
    """
    if progress_callback:
        try:
            progress_callback(0.05)
        except Exception:
            pass

    model = get_model()

    if progress_callback:
        try:
            progress_callback(0.2)
        except Exception:
            pass

    result = model.transcribe(file_path, fp16=False)

    if progress_callback:
        try:
            progress_callback(1.0)
        except Exception:
            pass

    return result["text"]
