"""
╔══════════════════════════════════════════════════════════╗
║   Darija → Français — Backend FastAPI                   ║
║   Whisper STT + Aya LLM (Ollama) + Coqui TTS           ║
╚══════════════════════════════════════════════════════════╝
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import whisper
import ollama
from TTS.api import TTS
import tempfile, os, uuid, shutil

# ── Config ──────────────────────────────────────────────
WHISPER_MODEL  = "large-v3"   # ou "medium" si GPU limité
OLLAMA_MODEL   = "aya:8b"     # ollama pull aya:8b
TTS_MODEL      = "tts_models/fr/mai/tacotron2-DDC"
AUDIO_OUT_DIR  = tempfile.mkdtemp()

print("⏳ Chargement Whisper...")
whisper_model = whisper.load_model(WHISPER_MODEL)
print(f"✅ Whisper {WHISPER_MODEL} prêt")

print("⏳ Chargement Coqui TTS...")
tts_engine = TTS(model_name=TTS_MODEL, progress_bar=False)
print("✅ Coqui TTS prêt")

# ── App ──────────────────────────────────────────────────
app = FastAPI(title="Darija Translator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # mobile sur WiFi local
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Modèles Pydantic ──────────────────────────────────────
class TranslateRequest(BaseModel):
    text: str

class TTSRequest(BaseModel):
    text: str

# ── Route: santé ──────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "Darija Translator API en ligne 🎙"}

# ── Route: STT (Whisper) ──────────────────────────────────
@app.post("/stt")
async def speech_to_text(audio: UploadFile = File(...)):
    """
    Reçoit un fichier audio (webm/wav/mp3),
    retourne la transcription en Darija via Whisper.
    """
    suffix = os.path.splitext(audio.filename or "audio.webm")[1] or ".webm"
    tmp_path = os.path.join(AUDIO_OUT_DIR, f"{uuid.uuid4()}{suffix}")

    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        result = whisper_model.transcribe(
            tmp_path,
            language="ar",
            # Prompt d'amorçage pour le Darija marocain
            initial_prompt="هذا نص بالدارجة المغربية، اللهجة المغربية الدارجة",
            task="transcribe",
            fp16=False,          # mettre True si GPU CUDA disponible
            temperature=0.0,
        )
        return {"text": result["text"].strip(), "language": result.get("language", "ar")}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur STT: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ── Route: Traduction (Aya via Ollama) ───────────────────
@app.post("/translate")
def translate(req: TranslateRequest):
    """
    Traduit un texte Darija marocain vers le français
    via le modèle Aya (Cohere) tournant sur Ollama en local.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Texte vide")

    prompt = (
        "Tu es un traducteur expert du dialecte marocain (Darija) vers le français. "
        "Traduis le texte suivant en français naturel et courant, sans explication ni commentaire. "
        "Retourne uniquement la traduction française.\n\n"
        f"Darija : {req.text}\n\n"
        "Traduction française :"
    )

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2, "num_predict": 300},
        )
        translation = response["message"]["content"].strip()
        # Nettoyer les éventuels préfixes parasites
        for prefix in ["Traduction française :", "Traduction :", "French:", "FR:"]:
            if translation.lower().startswith(prefix.lower()):
                translation = translation[len(prefix):].strip()

        return {"translation": translation, "original": req.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur LLM: {str(e)}")

# ── Route: TTS (Coqui) ───────────────────────────────────
@app.post("/tts")
def text_to_speech(req: TTSRequest):
    """
    Synthétise un texte français en audio WAV via Coqui TTS.
    Retourne le fichier audio directement.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Texte vide")

    out_path = os.path.join(AUDIO_OUT_DIR, f"{uuid.uuid4()}.wav")

    try:
        tts_engine.tts_to_file(text=req.text, file_path=out_path)
        return FileResponse(
            out_path,
            media_type="audio/wav",
            filename="traduction.wav",
            background=None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur TTS: {str(e)}")

# ── Route: Pipeline complet en une seule requête ──────────
@app.post("/pipeline")
async def full_pipeline(audio: UploadFile = File(...)):
    """
    Pipeline complet : audio Darija → texte Darija → traduction FR → audio FR
    Retourne JSON avec darija, french, et audio_url pour récupérer le son.
    """
    # 1. STT
    suffix = os.path.splitext(audio.filename or "audio.webm")[1] or ".webm"
    tmp_in  = os.path.join(AUDIO_OUT_DIR, f"{uuid.uuid4()}{suffix}")
    tmp_out = os.path.join(AUDIO_OUT_DIR, f"{uuid.uuid4()}.wav")

    try:
        with open(tmp_in, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        stt_result = whisper_model.transcribe(
            tmp_in, language="ar",
            initial_prompt="هذا نص بالدارجة المغربية",
            fp16=False,
        )
        darija = stt_result["text"].strip()

        # 2. Traduction
        prompt = (
            "Traduis ce texte Darija marocain en français naturel sans explication:\n\n"
            f"{darija}\n\nTraduction française :"
        )
        llm_resp = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
        )
        french = llm_resp["message"]["content"].strip()

        # 3. TTS
        tts_engine.tts_to_file(text=french, file_path=tmp_out)
        audio_id = os.path.basename(tmp_out)

        return JSONResponse({
            "darija": darija,
            "french": french,
            "audio_id": audio_id,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_in):
            os.remove(tmp_in)

@app.get("/audio/{audio_id}")
def get_audio(audio_id: str):
    """Sert le fichier audio généré par /pipeline."""
    path = os.path.join(AUDIO_OUT_DIR, audio_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Audio introuvable")
    return FileResponse(path, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    # Lance sur 0.0.0.0 pour être accessible depuis le téléphone (même WiFi)
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
