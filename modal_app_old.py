from modal import App, Image, asgi_app

app = App("real-time-voice-cloning-backend")

# Define the container image with necessary system packages, Python libraries, and local files
image = (
    Image.debian_slim(python_version="3.9")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir("encoder", remote_path="/root/encoder")
    .add_local_dir("synthesizer", remote_path="/root/synthesizer")
    .add_local_dir("vocoder", remote_path="/root/vocoder")
    .add_local_dir("utils", remote_path="/root/utils")
    .add_local_dir("saved_models", remote_path="/root/saved_models")
    .add_local_file("server.py", remote_path="/root/server.py")
)

# Create the ASGI app function, requesting a T4 GPU (fits nicely within a 16GB VRAM constraint)
@app.function(
    image=image,
    gpu="T4",
    timeout=600 # 10 minutes timeout
)
@asgi_app()
def fastapi_app():
    import sys
    sys.path.append("/root")
    
    # Import our existing FastAPI app
    from server import app as backend_app
    return backend_app














import os
import uuid
import io
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional

import modal
from fastapi import UploadFile, File, HTTPException, Form
from fastapi.responses import Response
from pydantic import BaseModel

# --- 1. Pydantic Models ---
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

# --- 2. Define the Container Image ---
app = modal.App("real-time-voice-cloning-backend")

image = (
    modal.Image.debian_slim(python_version="3.10") # Bumped to 3.10
    .apt_install("ffmpeg", "libsndfile1", "build-essential") # Keeping build-essential for webrtcvad
    .pip_install_from_requirements("requirements.txt")
    .workdir("/root")
    .add_local_dir("encoder", remote_path="/root/encoder")
    .add_local_dir("synthesizer", remote_path="/root/synthesizer")
    .add_local_dir("vocoder", remote_path="/root/vocoder")
    .add_local_dir("utils", remote_path="/root/utils")
    .add_local_dir("saved_models", remote_path="/root/saved_models")
)
# --- 3. The Main Inference Class ---
@app.cls(
    image=image, 
    gpu="T4", 
    timeout=600,
    container_idle_timeout=300, 
    allow_concurrent_inputs=10 
)
class VoiceCloningAPI:
    @modal.enter()
    def setup(self):
        import sys
        if "/root" not in sys.path:
            sys.path.append("/root")
            
        from encoder import inference as encoder
        from synthesizer.inference import Synthesizer
        from vocoder import inference as vocoder

        self.encoder = encoder
        self.Synthesizer = Synthesizer
        self.vocoder = vocoder
        self.synthesizer_instance = None
        self.cached_specs = {}

    # --- 4. The Routes ---

    @modal.web_endpoint(method="GET")
    def list_models(self):
        models_dir = Path("/root/saved_models")
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

    @modal.web_endpoint(method="POST")
    def load_models(self, req: LoadModelsRequest):
        try:
            if not Path(req.encoder_path).exists():
                raise HTTPException(status_code=400, detail="Encoder model not found")
            if not Path(req.synthesizer_path).exists():
                raise HTTPException(status_code=400, detail="Synthesizer model not found")
            
            self.encoder.load_model(Path(req.encoder_path))
            self.synthesizer_instance = self.Synthesizer(Path(req.synthesizer_path))
            
            if req.vocoder_path and req.vocoder_path != "Griffin-Lim":
                if not Path(req.vocoder_path).exists():
                    raise HTTPException(status_code=400, detail="Vocoder model not found")
                self.vocoder.load_model(Path(req.vocoder_path))
                
            return {"status": "success", "message": "Models loaded successfully"}
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @modal.web_endpoint(method="POST")
    async def extract_embedding(self, file: UploadFile = File(...)):
        import numpy as np # Delayed import
        
        if not self.encoder.is_loaded():
            raise HTTPException(status_code=400, detail="Encoder model not loaded")
            
        try:
            audio_bytes = await file.read()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
                
            try:
                wav = self.Synthesizer.load_preprocess_wav(tmp_path)
                encoder_wav = self.encoder.preprocess_wav(wav)
                embed, partial_embeds, _ = self.encoder.embed_utterance(encoder_wav, return_partials=True)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            
            return {"embed": embed.tolist()}
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @modal.web_endpoint(method="POST")
    def synthesize_spectrogram(self, req: SynthesizeRequest):
        import numpy as np # Delayed import
        import torch       # Delayed import
        
        if self.synthesizer_instance is None:
            raise HTTPException(status_code=400, detail="Synthesizer model not loaded")
            
        if req.seed is not None:
            torch.manual_seed(req.seed)
            
        try:
            texts = req.text.split("\n")
            embeds = [np.array(req.embed)] * len(texts)
            
            specs = self.synthesizer_instance.synthesize_spectrograms(texts, embeds)
            breaks = [spec.shape[1] for spec in specs]
            spec = np.concatenate(specs, axis=1)
            
            spec_id = str(uuid.uuid4())
            self.cached_specs[spec_id] = (spec, breaks)
            
            return {"spec_id": spec_id, "breaks": breaks}
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @modal.web_endpoint(method="POST")
    def vocode_audio(self, req: VocodeRequest):
        import numpy as np     # Delayed import
        import torch           # Delayed import
        import soundfile as sf # Delayed import
        
        if req.spec_id not in self.cached_specs:
            raise HTTPException(status_code=404, detail="Spectrogram not found")
            
        if req.seed is not None:
            torch.manual_seed(req.seed)
            
        try:
            spec, breaks = self.cached_specs[req.spec_id]
            
            if self.vocoder.is_loaded():
                wav = self.vocoder.infer_waveform(spec)
            else:
                wav = self.Synthesizer.griffin_lim(spec)
                
            b_ends = np.cumsum(np.array(breaks) * self.Synthesizer.hparams.hop_size)
            b_starts = np.concatenate(([0], b_ends[:-1]))
            wavs = [wav[start:end] for start, end in zip(b_starts, b_ends)]
            breaks_audio = [np.zeros(int(0.15 * self.Synthesizer.sample_rate))] * len(breaks)
            wav = np.concatenate([i for w, b in zip(wavs, breaks_audio) for i in (w, b)])
            
            wav = wav / np.abs(wav).max() * 0.97
            
            with io.BytesIO() as out_io:
                sf.write(out_io, wav, self.Synthesizer.sample_rate, format='WAV')
                out_bytes = out_io.getvalue()
                
            return Response(content=out_bytes, media_type="audio/wav")
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @modal.web_endpoint(method="POST")
    def generate_audio(self, req: GenerateRequest):
        import numpy as np     # Delayed import
        import torch           # Delayed import
        import soundfile as sf # Delayed import
        
        if self.synthesizer_instance is None:
            raise HTTPException(status_code=400, detail="Synthesizer model not loaded")
            
        if req.seed is not None:
            torch.manual_seed(req.seed)
            
        try:
            texts = req.text.split("\n")
            embeds = [np.array(req.embed)] * len(texts)
            specs = self.synthesizer_instance.synthesize_spectrograms(texts, embeds)
            breaks = [spec.shape[1] for spec in specs]
            spec = np.concatenate(specs, axis=1)
            
            if self.vocoder.is_loaded():
                wav = self.vocoder.infer_waveform(spec)
            else:
                wav = self.Synthesizer.griffin_lim(spec)
                
            b_ends = np.cumsum(np.array(breaks) * self.Synthesizer.hparams.hop_size)
            b_starts = np.concatenate(([0], b_ends[:-1]))
            wavs = [wav[start:end] for start, end in zip(b_starts, b_ends)]
            breaks_audio = [np.zeros(int(0.15 * self.Synthesizer.sample_rate))] * len(breaks)
            wav = np.concatenate([i for w, b in zip(wavs, breaks_audio) for i in (w, b)])
            
            wav = wav / np.abs(wav).max() * 0.97
            
            with io.BytesIO() as out_io:
                sf.write(out_io, wav, self.Synthesizer.sample_rate, format='WAV')
                out_bytes = out_io.getvalue()
                
            return Response(content=out_bytes, media_type="audio/wav")
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
        



import os
import uuid
import io
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional

import modal
from fastapi import UploadFile, File, HTTPException, Form
from fastapi.responses import Response
from pydantic import BaseModel

# --- 1. Pydantic Models ---
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

# --- 2. Infrastructure Setup (Docker + Volumes) ---
app = modal.App("real-time-voice-cloning-backend")

# Build the image cleanly from the Dockerfile
image = modal.Image.from_dockerfile("Dockerfile")

# Reference the persistent volume holding your heavy .pt files
model_volume = modal.Volume.from_name("voice-cloning-models")

# --- 3. The Main Inference Class ---
@app.cls(
    image=image, 
    gpu="T4", 
    volumes={"/root/saved_models": model_volume}, # Mounts the heavy weights here!
    timeout=600,
    scaledown_window=300, 
    allow_concurrent_inputs=10 
)
class VoiceCloningAPI:
    @modal.enter()
    def setup(self):
        import sys
        if "/root" not in sys.path:
            sys.path.append("/root")
            
        from encoder import inference as encoder
        from synthesizer.inference import Synthesizer
        from vocoder import inference as vocoder

        self.encoder = encoder
        self.Synthesizer = Synthesizer
        self.vocoder = vocoder
        self.synthesizer_instance = None
        self.cached_specs = {}

    # --- 4. The Complete Routes ---

    @modal.web_endpoint(method="GET")
    def list_models(self):
        # This will now look inside your mounted Volume
        models_dir = Path("/root/saved_models")
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

    @modal.web_endpoint(method="POST")
    def load_models(self, req: LoadModelsRequest):
        try:
            if not Path(req.encoder_path).exists():
                raise HTTPException(status_code=400, detail="Encoder model not found")
            if not Path(req.synthesizer_path).exists():
                raise HTTPException(status_code=400, detail="Synthesizer model not found")
            
            self.encoder.load_model(Path(req.encoder_path))
            self.synthesizer_instance = self.Synthesizer(Path(req.synthesizer_path))
            
            if req.vocoder_path and req.vocoder_path != "Griffin-Lim":
                if not Path(req.vocoder_path).exists():
                    raise HTTPException(status_code=400, detail="Vocoder model not found")
                self.vocoder.load_model(Path(req.vocoder_path))
                
            return {"status": "success", "message": "Models loaded successfully"}
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @modal.web_endpoint(method="POST")
    async def extract_embedding(self, file: UploadFile = File(...)):
        import numpy as np # Delayed import
        
        if not self.encoder.is_loaded():
            raise HTTPException(status_code=400, detail="Encoder model not loaded")
            
        try:
            audio_bytes = await file.read()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
                
            try:
                wav = self.Synthesizer.load_preprocess_wav(tmp_path)
                encoder_wav = self.encoder.preprocess_wav(wav)
                embed, partial_embeds, _ = self.encoder.embed_utterance(encoder_wav, return_partials=True)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            
            return {"embed": embed.tolist()}
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @modal.web_endpoint(method="POST")
    def synthesize_spectrogram(self, req: SynthesizeRequest):
        import numpy as np # Delayed import
        import torch       # Delayed import
        
        if self.synthesizer_instance is None:
            raise HTTPException(status_code=400, detail="Synthesizer model not loaded")
            
        if req.seed is not None:
            torch.manual_seed(req.seed)
            
        try:
            texts = req.text.split("\n")
            embeds = [np.array(req.embed)] * len(texts)
            
            specs = self.synthesizer_instance.synthesize_spectrograms(texts, embeds)
            breaks = [spec.shape[1] for spec in specs]
            spec = np.concatenate(specs, axis=1)
            
            spec_id = str(uuid.uuid4())
            self.cached_specs[spec_id] = (spec, breaks)
            
            return {"spec_id": spec_id, "breaks": breaks}
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @modal.web_endpoint(method="POST")
    def vocode_audio(self, req: VocodeRequest):
        import numpy as np     # Delayed import
        import torch           # Delayed import
        import soundfile as sf # Delayed import
        
        if req.spec_id not in self.cached_specs:
            raise HTTPException(status_code=404, detail="Spectrogram not found")
            
        if req.seed is not None:
            torch.manual_seed(req.seed)
            
        try:
            spec, breaks = self.cached_specs[req.spec_id]
            
            if self.vocoder.is_loaded():
                wav = self.vocoder.infer_waveform(spec)
            else:
                wav = self.Synthesizer.griffin_lim(spec)
                
            b_ends = np.cumsum(np.array(breaks) * self.Synthesizer.hparams.hop_size)
            b_starts = np.concatenate(([0], b_ends[:-1]))
            wavs = [wav[start:end] for start, end in zip(b_starts, b_ends)]
            breaks_audio = [np.zeros(int(0.15 * self.Synthesizer.sample_rate))] * len(breaks)
            wav = np.concatenate([i for w, b in zip(wavs, breaks_audio) for i in (w, b)])
            
            wav = wav / np.abs(wav).max() * 0.97
            
            with io.BytesIO() as out_io:
                sf.write(out_io, wav, self.Synthesizer.sample_rate, format='WAV')
                out_bytes = out_io.getvalue()
                
            return Response(content=out_bytes, media_type="audio/wav")
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @modal.web_endpoint(method="POST")
    def generate_audio(self, req: GenerateRequest):
        import numpy as np     # Delayed import
        import torch           # Delayed import
        import soundfile as sf # Delayed import
        
        if self.synthesizer_instance is None:
            raise HTTPException(status_code=400, detail="Synthesizer model not loaded")
            
        if req.seed is not None:
            torch.manual_seed(req.seed)
            
        try:
            texts = req.text.split("\n")
            embeds = [np.array(req.embed)] * len(texts)
            specs = self.synthesizer_instance.synthesize_spectrograms(texts, embeds)
            breaks = [spec.shape[1] for spec in specs]
            spec = np.concatenate(specs, axis=1)
            
            if self.vocoder.is_loaded():
                wav = self.vocoder.infer_waveform(spec)
            else:
                wav = self.Synthesizer.griffin_lim(spec)
                
            b_ends = np.cumsum(np.array(breaks) * self.Synthesizer.hparams.hop_size)
            b_starts = np.concatenate(([0], b_ends[:-1]))
            wavs = [wav[start:end] for start, end in zip(b_starts, b_ends)]
            breaks_audio = [np.zeros(int(0.15 * self.Synthesizer.sample_rate))] * len(breaks)
            wav = np.concatenate([i for w, b in zip(wavs, breaks_audio) for i in (w, b)])
            
            wav = wav / np.abs(wav).max() * 0.97
            
            with io.BytesIO() as out_io:
                sf.write(out_io, wav, self.Synthesizer.sample_rate, format='WAV')
                out_bytes = out_io.getvalue()
                
            return Response(content=out_bytes, media_type="audio/wav")
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))