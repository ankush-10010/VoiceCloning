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