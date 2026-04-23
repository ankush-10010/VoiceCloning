import modal
import subprocess
import time
import httpx
from fastapi import UploadFile, File, Response

app = modal.App("real-time-voice-cloning-proxy")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "curl")
    .pip_install("fastapi[standard]")
    .add_local_file("requirements.txt", "/root/requirements.txt", copy=True)
    
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh")
    
    .run_commands(
        "uv venv --python 3.9 /root/venv",
        "uv pip install --python /root/venv 'setuptools<70.0.0' wheel",
        
        # CRITICAL FIX: Explicitly install PyTorch before your requirements.txt
        "uv pip install --python /root/venv torch torchaudio",
        
        "uv pip install --python /root/venv --no-build-isolation -r /root/requirements.txt",
        # "uv pip install --python /root/venv uvicorn fastapi python-multipart" 

        "uv pip install --python /root/venv uvicorn fastapi python-multipart matplotlib"
    )
    .add_local_dir(".", remote_path="/root", ignore=["saved_models/**", ".venv/**", "__pycache__/**"])
)

model_volume = modal.Volume.from_name("voice-cloning-models")

@app.cls(
    image=image, 
    gpu="T4", 
    volumes={"/root/saved_models": model_volume},
    timeout=600,
    scaledown_window=300
    # Removed allow_concurrent_inputs to fix the deprecation warning
)
class VoiceCloningProxy:
    @modal.enter()
    def setup(self):
        import socket
        
        print("Starting Python 3.9 background server...")
        self.process = subprocess.Popen(
            ["/root/venv/bin/python", "-m", "uvicorn", "api_39:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd="/root"
        )
        
        # Smart Polling: Wait until the port is actively accepting connections
        start_time = time.time()
        is_ready = False
        print("Waiting for Uvicorn to bind to port 8000...")
        
        while time.time() - start_time < 90:  # Give it up to 90 seconds to load heavy models
            try:
                with socket.create_connection(("127.0.0.1", 8000), timeout=1):
                    is_ready = True
                    break
            except OSError:
                time.sleep(1)
                
        if is_ready:
            print("✅ Inner Python 3.9 Server is UP and ready!")
        else:
            print("❌ Inner Server FAILED to start within 90 seconds.")

    # --- Proxy Routes (Updated to fastapi_endpoint) ---
    
    # @modal.fastapi_endpoint(method="GET")
    # async def list_models(self):
    #     async with httpx.AsyncClient() as client:
    #         response = await client.get("http://127.0.0.1:8000/list_models", timeout=30)
    #         return response.json()

    @modal.fastapi_endpoint(method="POST")
    async def load_models(self, req: dict):
        async with httpx.AsyncClient() as client:
            response = await client.post("http://127.0.0.1:8000/load_models", json=req, timeout=120)
            return response.json()

    @modal.fastapi_endpoint(method="POST")
    async def extract_embedding(self, file: UploadFile = File(...)):
        file_bytes = await file.read()
        files = {'file': (file.filename, file_bytes, file.content_type)}
        async with httpx.AsyncClient() as client:
            response = await client.post("http://127.0.0.1:8000/extract_embedding", files=files, timeout=120)
            return response.json()

    @modal.fastapi_endpoint(method="POST")
    async def synthesize_spectrogram(self, req: dict):
        async with httpx.AsyncClient() as client:
            response = await client.post("http://127.0.0.1:8000/synthesize_spectrogram", json=req, timeout=120)
            return response.json()

    @modal.fastapi_endpoint(method="POST")
    async def vocode_audio(self, req: dict):
        async with httpx.AsyncClient() as client:
            response = await client.post("http://127.0.0.1:8000/vocode_audio", json=req, timeout=120)
            return Response(content=response.content, media_type="audio/wav")

    @modal.fastapi_endpoint(method="POST")
    async def generate_audio(self, req: dict):
        async with httpx.AsyncClient() as client:
            response = await client.post("http://127.0.0.1:8000/generate_audio", json=req, timeout=300)
            return Response(content=response.content, media_type="audio/wav")
        
    @modal.fastapi_endpoint(method="GET")
    async def spectrogram_image(self, spec_id: str):
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://127.0.0.1:8000/spectrogram_image/{spec_id}", timeout=60)
            return Response(content=response.content, media_type="image/png")