import os
import sys
import io
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import edge_tts

sys.stdout.reconfigure(encoding='utf-8')

app = FastAPI(title="BhashaSetu ML Translation & TTS API")

# Enable CORS for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "Hindi_Mundari_MT5"))
device = "cpu"
tokenizer = None
model = None

def get_system_ram_mb() -> float:
    """Detect available container RAM (Docker cgroup v1/v2 or OS sysconf)."""
    # Docker cgroup v2
    try:
        if os.path.exists("/sys/fs/cgroup/memory.max"):
            with open("/sys/fs/cgroup/memory.max", "r") as f:
                val = f.read().strip()
                if val.isdigit():
                    return int(val) / (1024 * 1024)
    except Exception:
        pass
    # Docker cgroup v1
    try:
        if os.path.exists("/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r") as f:
                val = f.read().strip()
                if val.isdigit():
                    limit = int(val)
                    if limit < (1024 * 1024 * 1024 * 1024):
                        return limit / (1024 * 1024)
    except Exception:
        pass
    # Linux sysconf
    try:
        pages = os.sysconf('SC_PHYS_PAGES')
        page_size = os.sysconf('SC_PAGE_SIZE')
        return (pages * page_size) / (1024 * 1024)
    except Exception:
        pass
    return 4096.0

def init_model():
    """Load model if sufficient RAM (>= 1.5GB) and weight files exist; otherwise stay in lightweight mode."""
    global tokenizer, model, device
    force_load = os.environ.get("FORCE_LOAD_MODEL", "false").lower() == "true"
    ram_mb = get_system_ram_mb()

    if ram_mb < 1500 and not force_load:
        print(f"[Resource Guard] Available RAM is ~{int(ram_mb)}MB (e.g. Render Free Tier 512MB).")
        print("[Resource Guard] Skipping 1.2GB mT5 in-memory model to avoid Out-Of-Memory kill.")
        print("[Resource Guard] Server running in lightweight mode with Neural TTS & API fallback.")
        return

    # Check for weights file before loading PyTorch
    weights_found = any(
        os.path.exists(os.path.join(MODEL_DIR, fname))
        for fname in ["model.safetensors", "pytorch_model.bin"]
    )
    if not weights_found and not force_load:
        print(f"[Model Loader] No weights found in '{MODEL_DIR}'. Running in lightweight fallback mode.")
        return

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Model Loader] Loading mT5 model on {device} from {MODEL_DIR}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device)
        model.eval()
        print("[Model Loader] mT5 model loaded successfully and ready for inference!")
    except Exception as e:
        print(f"[Model Loader] Could not load model: {e}")
        tokenizer = None
        model = None

# Initialize on boot
init_model()


# Ol Chiki -> Phonetic Devanagari transliteration map for Santhali speech
OL_CHIKI_VOWELS = {
    'ᱚ': ('अ', ''),
    'ᱟ': ('आ', 'ा'),
    'ᱤ': ('इ', 'ि'),
    'ᱩ': ('उ', 'ु'),
    'ᱮ': ('ए', 'े'),
    'ᱳ': ('ओ', 'ो'),
}

OL_CHIKI_CONSONANTS = {
    'ᱛ': 'त', 'ᱜ': 'ग', 'ᱝ': 'ंग', 'ᱞ': 'ल',
    'ᱠ': 'क', 'ᱡ': 'ज', 'ᱢ': 'म', 'ᱣ': 'व',
    'ᱥ': 'स', 'ᱦ': 'ह', 'ᱧ': 'ञ', 'ᱨ': 'र',
    'ᱪ': 'च', 'ᱫ': 'द', 'ᱬ': 'ण', 'ᱭ': 'य',
    'ᱯ': 'प', 'ᱰ': 'ड', 'ᱱ': 'न', 'ᱲ': 'ड़',
    'ᱴ': 'ट', 'ᱵ': 'ब', 'ᱶ': 'व', 'ᱷ': 'ह',
}

def ol_chiki_to_devanagari(text: str) -> str:
    """Converts Ol Chiki text into phonetic Devanagari for speech synthesis."""
    out = []
    prev_consonant = False
    for char in text:
        if char in OL_CHIKI_VOWELS:
            v_init, v_matra = OL_CHIKI_VOWELS[char]
            if prev_consonant:
                out.append(v_matra)
            else:
                out.append(v_init)
            prev_consonant = False
        elif char in OL_CHIKI_CONSONANTS:
            out.append(OL_CHIKI_CONSONANTS[char])
            prev_consonant = True
        elif char in ('ᱸ', 'ᱺ'):
            out.append('ँ')
            prev_consonant = False
        elif char == 'ᱽ':
            out.append('्')
            prev_consonant = False
        elif char == '᱾':
            out.append('।')
            prev_consonant = False
        elif char == '᱿':
            out.append('॥')
            prev_consonant = False
        else:
            out.append(char)
            prev_consonant = False
    return ''.join(out)

class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "hi"
    target_lang: str = "mundari"
    max_length: int = 128

class TranslationResponse(BaseModel):
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    phonetic_text: str
    meaning: str = ""

class TTSRequest(BaseModel):
    text: str
    voice: str = "hi-IN-SwaraNeural"
    rate: str = "+0%"

@app.get("/")
def root():
    return {
        "name": "BhashaSetu Backend API",
        "status": "online",
        "model_loaded": model is not None,
        "endpoints": {
            "status": "/api/status",
            "translate": "/api/translate",
            "tts": "/api/tts",
            "docs": "/docs"
        }
    }

@app.get("/api/status")
def status():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "mode": "full_ml_model" if model is not None else "lightweight_mode",
        "system_ram_mb": int(get_system_ram_mb()),
        "device": device,
        "features": ["translation", "neural-tts", "ol-chiki-phonetics"],
        "vocab_size": tokenizer.vocab_size if tokenizer is not None else 0
    }

@app.post("/api/translate", response_model=TranslationResponse)
def translate(req: TranslationRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    clean_text = req.text.strip()

    if model is None or tokenizer is None:
        FALLBACK_DICT = {
            "नमस्ते": "ᱡᱚᱦᱟᱨ",
            "पानी": "ᱫᱟᱜ",
            "फूल": "ᱵᱟᱦᱟ",
            "पेड़ हमें छाया देते हैं।": "ᱫᱟᱨᱮ ᱟᱵᱚ ᱩᱢᱩᱞ ᱮᱢᱟᱵᱚᱱᱟ᱾",
            "बच्चों, आज हम पेड़ के बारे में सीखेंगे।": "ᱜᱤᱫᱽᱨᱟᱹ, ᱛᱮᱦᱮᱧ ᱟᱵᱚ ᱫᱟᱨᱮ ᱵᱟᱵᱚᱛ ᱛᱮᱵᱚ ᱪᱮᱫᱚᱜᱼᱟ᱾",
        }
        translated = FALLBACK_DICT.get(clean_text, f"[Model weights pending on cloud: {clean_text}]")
        phonetic = ol_chiki_to_devanagari(translated) if any('\u1C50' <= c <= '\u1C7F' for c in translated) else translated
        return TranslationResponse(
            source_text=clean_text,
            translated_text=translated,
            source_lang=req.source_lang,
            target_lang=req.target_lang,
            phonetic_text=phonetic,
            meaning=f"Translation: {translated}"
        )

    try:
        import torch
        inputs = tokenizer(clean_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=req.max_length,
                num_beams=3,
                early_stopping=True
            )
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        phonetic = ol_chiki_to_devanagari(translated) if any('\u1C50' <= c <= '\u1C7F' for c in translated) else translated
        
        return TranslationResponse(
            source_text=clean_text,
            translated_text=translated,
            source_lang=req.source_lang,
            target_lang=req.target_lang,
            phonetic_text=phonetic,
            meaning=f"Translation: {translated}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tts")
async def text_to_speech(req: TTSRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Transliterate Ol Chiki if present so the Indian voice reads the Santhali words accurately
    spoken_text = ol_chiki_to_devanagari(req.text.strip()) if any('\u1C50' <= c <= '\u1C7F' for c in req.text) else req.text.strip()
    
    try:
        communicate = edge_tts.Communicate(spoken_text, voice=req.voice, rate=req.rate)
        audio_stream = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_stream.write(chunk["data"])
        
        audio_bytes = audio_stream.getvalue()
        if not audio_bytes:
            raise ValueError("No audio was generated")
            
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS synthesis error: {str(e)}")

from fastapi.staticfiles import StaticFiles

# Serve static frontend (works both locally and in Hugging Face Space)
dist_dir = os.path.join(os.path.dirname(__file__), "dist")
if not os.path.exists(dist_dir):
    dist_dir = os.path.join(os.path.dirname(__file__), "BhashaSetu", "dist")

if os.path.exists(dist_dir):
    app.mount("/", StaticFiles(directory=dist_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
