import subprocess
import queue
import sys
import os
import time
import sounddevice as sd
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel

# === Configuración ===
LANG = "es"
WHISPER_MODEL_SIZE = "small"  # opciones: tiny, base, small, medium
VOICE = "Monica"              # macOS: "Monica" o "Jorge" (español)
SAMPLE_RATE = 16000
CHANNELS = 1
MAX_SECONDS = 12              # máx. duración por turno
SILENCE_DB = -38              # umbral de silencio (dBFS aprox)
SILENCE_MIN_SEC = 0.8         # silencio continuo para cortar

# Carga del modelo STT (usa Metal)
model = WhisperModel(WHISPER_MODEL_SIZE, device="auto", compute_type="int8")

def rms_db(x: np.ndarray) -> float:
    x = x.astype(np.float32)
    if len(x) == 0:
        return -120.0
    rms = np.sqrt(np.mean(np.square(x)))
    if rms <= 1e-12:
        return -120.0
    return 20 * np.log10(rms + 1e-12)

def grabar_voz(filename="input.wav"):
    q = queue.Queue()
    rec = []

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    print("🎙️  Habla ahora... (grabando)")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback):
        t0 = time.time()
        silence_streak = 0.0
        while True:
            try:
                chunk = q.get(timeout=0.5)
            except queue.Empty:
                chunk = np.zeros((int(SAMPLE_RATE*0.1), CHANNELS), dtype=np.float32)

            rec.append(chunk)

            # Detección simple de silencio
            level = rms_db(chunk)
            if level < SILENCE_DB:
                silence_streak += 0.1
            else:
                silence_streak = 0.0

            dur = time.time() - t0
            if silence_streak >= SILENCE_MIN_SEC or dur >= MAX_SECONDS:
                break

    audio = np.concatenate(rec, axis=0)
    sf.write(filename, audio, SAMPLE_RATE)
    return filename

def transcribir(filename="input.wav"):
    segments, info = model.transcribe(filename, language=LANG, vad_filter=True, beam_size=1)
    text = "".join(seg.text for seg in segments).strip()
    return text

def consultar_ollama(prompt: str, model_name="llama3") -> str:
    # Llamada simple (bloqueante) a ollama
    res = subprocess.run(
        ["ollama", "run", model_name, prompt],
        capture_output=True, text=True
    )
    return res.stdout.strip()

def hablar(texto: str):
    # macOS TTS local
    subprocess.run(["say", "-v", VOICE, texto])

def main():
    print("🧠 Asistente de voz local (Ollama + Whisper + say)")
    print("Comandos: Enter para hablar, /salir para terminar\n")

    historial = []
    # Prompt del sistema (opcional)
    sistema = ("Eres un asistente conversacional local que responde en español de forma breve y clara. "
               "Si el usuario saluda, responde cordialmente.")

    while True:
        cmd = input("Presiona Enter para hablar (/salir para terminar): ").strip().lower()
        if cmd == "/salir":
            print("Hasta luego 👋")
            break

        # 1) Grabar
        wav = grabar_voz()

        # 2) STT
        user_text = transcribir(wav)
        if not user_text:
            print("No se detectó voz. Intentemos otra vez.")
            continue
        print(f"👤 Tú: {user_text}")

        # 3) Construir prompt con breve contexto
        contexto = ""
        for u, a in historial[-3:]:
            contexto += f"Usuario: {u}\nAsistente: {a}\n"
        prompt_llm = f"{sistema}\n\n{contexto}Usuario: {user_text}\nAsistente:"

        # 4) LLM (Ollama)
        answer = consultar_ollama(prompt_llm)
        # Limpieza