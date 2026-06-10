FROM runpod/base:0.6.2-cuda12.1.0

# Flux 1 Dev — v1
# Requires env vars:
#   SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
#   HF_TOKEN — HuggingFace token for gated model access
#   FLUX_BUCKET (optional) — default "flux-outputs"

RUN apt-get update && apt-get install -y python3.10 git curl ffmpeg && \
    curl https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    apt-get clean

RUN pip3.10 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip3.10 install diffusers==0.31.0 transformers==4.50.0 accelerate \
        sentencepiece protobuf huggingface_hub \
        optimum runpod supabase Pillow && \
    pip3.10 install "click>=8.0"

RUN mkdir -p /app
COPY handler.py /app/handler.py

CMD ["python3.10", "/app/handler.py"]
