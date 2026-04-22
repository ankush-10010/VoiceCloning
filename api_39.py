import os
import uuid
import io
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional
import sys

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import numpy as np
import torch
import soundfile as sf

import matplotlib
# Force matplotlib to not use any Xwindows backend so it doesn't crash on headless servers
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from fastapi.responses import Response, StreamingResponse

# Ensure the local modules can be found
if "/root" not in sys.path:
    sys.path.append("/root")

from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder

app = FastAPI()

# Global state to hold loaded models
synthesizer_instance = None
cached_specs = {}

# --- Pydantic Models ---
class LoadModelsRequest(BaseModel):
    encoder_path: str
    synthesizer_path: str
    vocoder_path: Optional[str] = None

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
@app.get("/list_models")
def list_models():
    models_dir = Path("/root/saved_models")
    if not models_dir.exists():
        return {"encoders": [], "synthesizers": [], "vocoders": []}

    return {
        "encoders": [str(f) for f in models_dir.glob("*/encoder.pt")],
        "synthesizers": [str(f) for f in models_dir.glob("*/synthesizer.pt")],
        "vocoders": [str(f) for f in models_dir.glob("*/vocoder.pt")]
    }

@app.post("/load_models")
def load_models(req: LoadModelsRequest):
    global synthesizer_instance
    try:
        encoder.load_model(Path(req.encoder_path))
        synthesizer_instance = Synthesizer(Path(req.synthesizer_path))
        
        if req.vocoder_path and req.vocoder_path != "Griffin-Lim":
            vocoder.load_model(Path(req.vocoder_path))
            
        return {"status": "success", "message": "Models loaded successfully"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract_embedding")
async def extract_embedding(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
            
        try:
            wav = Synthesizer.load_preprocess_wav(tmp_path)
            encoder_wav = encoder.preprocess_wav(wav)
            embed, _, _ = encoder.embed_utterance(encoder_wav, return_partials=True)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        return {"embed": embed.tolist()}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/synthesize_spectrogram")
def synthesize_spectrogram(req: SynthesizeRequest):
    global synthesizer_instance
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

@app.post("/vocode_audio")
def vocode_audio(req: VocodeRequest):
    if req.seed is not None:
        torch.manual_seed(req.seed)
        
    try:
        spec, breaks = cached_specs[req.spec_id]
        
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

@app.post("/generate_audio")
def generate_audio(req: GenerateRequest):
    global synthesizer_instance
    if req.seed is not None:
        torch.manual_seed(req.seed)
        
    try:
        texts = req.text.split("\n")
        embeds = [np.array(req.embed)] * len(texts)
        specs = synthesizer_instance.synthesize_spectrograms(texts, embeds)
        breaks = [spec.shape[1] for spec in specs]
        spec = np.concatenate(specs, axis=1)
        
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

    


# ... (existing code) ...

@app.get("/spectrogram_image/{spec_id}")
def get_spectrogram_image(spec_id: str):
    if spec_id not in cached_specs:
        raise HTTPException(status_code=404, detail="Spectrogram not found in cache.")
        
    try:
        spec, _ = cached_specs[spec_id]
        
        # Plotting logic adapted from ui.py -> draw_spec()
        fig, ax = plt.subplots(figsize=(10, 2.25))
        ax.imshow(spec, aspect="auto", interpolation="none")
        ax.set_title("Mel Spectrogram")
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Save the plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', facecolor="#F0F0F0")
        buf.seek(0)
        plt.close(fig) # Free memory!
        
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))