# ml/tts_provider.py
import base64, tempfile, os
import numpy as np
import soundfile as sf

class TTSProvider:
    def tts(self, text: str) -> dict:
        raise NotImplementedError

def _trim_silence_int16(mono: np.ndarray, thresh=500, pad=200):
    """
    Trim leading/trailing near-silence from int16 PCM.
    thresh=500 is ~1.5% full scale. pad (samples) keeps tiny context.
    """
    if mono.size == 0:
        return mono
    idx = np.where(np.abs(mono) > thresh)[0]
    if idx.size == 0:
        return mono
    start = max(int(idx[0]) - pad, 0)
    end   = min(int(idx[-1]) + pad, mono.shape[0]-1)
    return mono[start:end]

class Pyttsx3Provider(TTSProvider):
    def __init__(self, voice_id: str | None = None, rate: int | None = 175):
        try:
            import pyttsx3
            self.engine = pyttsx3.init()  # Windows SAPI5 by default
            if rate is not None:
                self.engine.setProperty("rate", rate)
            if voice_id is not None:
                self.engine.setProperty("voice", voice_id)
            # Uncomment to list available voices once:
            # for v in self.engine.getProperty("voices"):
            #     print("VOICE:", v.id)
        except Exception as e:
            print("[warn] pyttsx3 init failed:", e)
            self.engine = None

    def tts(self, text: str) -> dict:
        if self.engine is None:
            return {"error": "TTS unavailable"}
        # synthesize to a temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        self.engine.save_to_file(text, wav_path)
        self.engine.runAndWait()

        # read WAV -> mono int16
        data, sr = sf.read(wav_path, dtype="int16", always_2d=True)
        try:
            os.remove(wav_path)
        except OSError:
            pass

        mono = data.mean(axis=1).astype(np.int16)
        mono = _trim_silence_int16(mono)  # reduce initial silence

        b64 = base64.b64encode(mono.tobytes()).decode("ascii")
        return {"sample_rate": int(sr), "channels": 1, "pcm16": b64}
