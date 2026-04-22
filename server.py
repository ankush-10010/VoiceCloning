import os
import uuid
import io
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder

app = FastAPI(title="SV2TTS Backend API")

# Allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances and caches
synthesizer_instance = None
cached_specs = {}

# --- Pydantic Models ---

class LoadModelsRequest(BaseModel):
    encoder_path: str
    synthesizer_path: str
    vocoder_path: Optional[str] = None # None means Griffin-Lim

class SynthesizeRequest(BaseModel):
    text: str
    embed: List[float]
    seed: Optional[int] = None

class VocodeRequest(BaseModel):
    spec_id: str
    seed: Optional[int] = None

class GenerateRequest(BaseModel):
    text: str
    embed: List[float]
    seed: Optional[int] = None

# --- Routes ---

@app.get("/api/models")
def list_models():
    models_dir = Path("saved_models")
    if not models_dir.exists():
        return {"encoders": [], "synthesizers": [], "vocoders": []}

    encoders = [str(f) for f in models_dir.glob("*/encoder.pt")]
    synthesizers = [str(f) for f in models_dir.glob("*/synthesizer.pt")]
    vocoders = [str(f) for f in models_dir.glob("*/vocoder.pt")]

    return {
        "encoders": encoders,
        "synthesizers": synthesizers,
        "vocoders": vocoders
    }

@app.post("/api/models/load")
def load_models(req: LoadModelsRequest):
    global synthesizer_instance
    try:
        if not Path(req.encoder_path).exists():
            raise HTTPException(status_code=400, detail="Encoder model not found")
        if not Path(req.synthesizer_path).exists():
            raise HTTPException(status_code=400, detail="Synthesizer model not found")
        
        encoder.load_model(Path(req.encoder_path))
        synthesizer_instance = Synthesizer(Path(req.synthesizer_path))
        
        if req.vocoder_path and req.vocoder_path != "Griffin-Lim":
            if not Path(req.vocoder_path).exists():
                raise HTTPException(status_code=400, detail="Vocoder model not found")
            vocoder.load_model(Path(req.vocoder_path))
            
        return {"status": "success", "message": "Models loaded successfully"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/encoder/embed")
async def extract_embedding(file: UploadFile = File(...)):
    if not encoder.is_loaded():
        raise HTTPException(status_code=400, detail="Encoder model not loaded")
        
    try:
        audio_bytes = await file.read()
        
        # Save temporarily to let soundfile/librosa parse it from a real file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
            
        try:
            # Use the provided preprocess logic which requires a string path
            wav = Synthesizer.load_preprocess_wav(tmp_path)
            encoder_wav = encoder.preprocess_wav(wav)
            embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        return {"embed": embed.tolist()}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/synthesizer/synthesize")
def synthesize_spectrogram(req: SynthesizeRequest):
    global synthesizer_instance
    if synthesizer_instance is None:
        raise HTTPException(status_code=400, detail="Synthesizer model not loaded")
        
    if req.seed is not None:
        torch.manual_seed(req.seed)
        
    try:
        texts = req.text.split("\n")
        embeds = [np.array(req.embed)] * len(texts)
        
        specs = synthesizer_instance.synthesize_spectrograms(texts, embeds)
        breaks = [spec.shape[1] for spec in specs]
        spec = np.concatenate(specs, axis=1)
        
        spec_id = str(uuid.uuid4())
        cached_specs[spec_id] = (spec, breaks)
        
        return {"spec_id": spec_id, "breaks": breaks}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vocoder/vocode")
def vocode_audio(req: VocodeRequest):
    if req.spec_id not in cached_specs:
        raise HTTPException(status_code=404, detail="Spectrogram not found")
        
    if req.seed is not None:
        torch.manual_seed(req.seed)
        
    try:
        spec, breaks = cached_specs[req.spec_id]
        
        if vocoder.is_loaded():
            wav = vocoder.infer_waveform(spec)
        else:
            # Fallback to Griffin-Lim
            wav = Synthesizer.griffin_lim(spec)
            
        # Add breaks
        b_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)
        b_starts = np.concatenate(([0], b_ends[:-1]))
        wavs = [wav[start:end] for start, end in zip(b_starts, b_ends)]
        breaks_audio = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
        wav = np.concatenate([i for w, b in zip(wavs, breaks_audio) for i in (w, b)])
        
        # Normalize
        wav = wav / np.abs(wav).max() * 0.97
        
        # Convert to bytes
        with io.BytesIO() as out_io:
            sf.write(out_io, wav, Synthesizer.sample_rate, format='WAV')
            out_bytes = out_io.getvalue()
            
        return Response(content=out_bytes, media_type="audio/wav")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate")
def generate_audio(req: GenerateRequest):
    global synthesizer_instance
    if synthesizer_instance is None:
        raise HTTPException(status_code=400, detail="Synthesizer model not loaded")
        
    if req.seed is not None:
        torch.manual_seed(req.seed)
        
    try:
        # 1. Synthesize
        texts = req.text.split("\n")
        embeds = [np.array(req.embed)] * len(texts)
        specs = synthesizer_instance.synthesize_spectrograms(texts, embeds)
        breaks = [spec.shape[1] for spec in specs]
        spec = np.concatenate(specs, axis=1)
        
        # 2. Vocode
        if vocoder.is_loaded():
            wav = vocoder.infer_waveform(spec)
        else:
            wav = Synthesizer.griffin_lim(spec)
            
        b_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)
        b_starts = np.concatenate(([0], b_ends[:-1]))
        wavs = [wav[start:end] for start, end in zip(b_starts, b_ends)]
        breaks_audio = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
        wav = np.concatenate([i for w, b in zip(wavs, breaks_audio) for i in (w, b)])
        
        wav = wav / np.abs(wav).max() * 0.97
        
        with io.BytesIO() as out_io:
            sf.write(out_io, wav, Synthesizer.sample_rate, format='WAV')
            out_bytes = out_io.getvalue()
            
        return Response(content=out_bytes, media_type="audio/wav")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
