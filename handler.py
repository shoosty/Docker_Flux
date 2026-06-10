
"""Flux 1 Dev — v1 — RunPod Serverless handler.

Generates images from text prompts using Flux 1 Dev.
Uploads to Supabase and returns a public URL.

Required env vars:
  SUPABASE_URL              — https://<project>.supabase.co
  SUPABASE_SERVICE_ROLE_KEY — service-role JWT
  HF_TOKEN                  — HuggingFace token for gated model access
  FLUX_BUCKET (optional)    — default "flux-outputs"

CRITICAL: RunPod serverless mounts network volumes at /runpod-volume,
NOT /workspace. Every path here uses /runpod-volume/.
"""
import runpod
import os
import uuid
import time
import traceback
import tempfile

import torch
from diffusers import FluxPipeline
from PIL import Image
from supabase import create_client

# ── Model path ────────────────────────────────────────────────────
MODEL_PATH = "/runpod-volume/models/flux-1-dev"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Models not found at {MODEL_PATH} - check volume mount!")

print(f"Models found at {MODEL_PATH}")

print("Loading Flux 1 Dev pipeline...")
pipe = FluxPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
print("Pipeline loaded!")

# ── Supabase client ────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
FLUX_BUCKET = os.environ.get("FLUX_BUCKET", "flux-outputs")

_supabase_client = None
def supabase_client():
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set."
        )
    _supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    return _supabase_client

def upload_to_supabase(local_path: str, storage_path: str, content_type: str) -> str:
    sb = supabase_client()
    with open(local_path, "rb") as f:
        data = f.read()
    sb.storage.from_(FLUX_BUCKET).upload(
        path=storage_path,
        file=data,
        file_options={"content-type": content_type, "upsert": "true"},
    )
    res = sb.storage.from_(FLUX_BUCKET).get_public_url(storage_path)
    if isinstance(res, dict):
        return res.get("publicUrl") or res.get("publicURL") or res.get("public_url")
    return res

def handler(job):
    """RunPod serverless entrypoint.

    Inputs:
      prompt (str)           — text prompt. Required.
      width (int)            — image width. Default 1024.
      height (int)           — image height. Default 1024.
      num_inference_steps (int) — Default 28.
      guidance_scale (float) — Default 3.5.
      seed (int)             — for reproducibility. Optional.
      format (str)           — "png" (default) or "jpg".
      storage_path (str)     — explicit object path. Optional.

    Output (success):
      {
        "image_url":    "<public URL>",
        "format":       "png" | "jpg",
        "storage_path": "<path>",
        "width":        <int>,
        "height":       <int>,
        "seed":         <int>
      }

    Output (failure):
      { "error": "<message>", "traceback": "<full trace>" }
    """
    try:
        inp = job.get("input", {}) or {}

        prompt  = inp.get("prompt")
        if not prompt:
            return {"error": "prompt is required"}

        width    = int(inp.get("width", 1024))
        height   = int(inp.get("height", 1024))
        steps    = int(inp.get("num_inference_steps", 28))
        guidance = float(inp.get("guidance_scale", 3.5))
        fmt      = (inp.get("format") or "png").lower()
        if fmt not in ("png", "jpg"):
            return {"error": f"format must be 'png' or 'jpg', got '{fmt}'"}

        seed = inp.get("seed")
        if seed is not None:
            generator = torch.Generator().manual_seed(int(seed))
        else:
            seed = torch.randint(0, 2**32, (1,)).item()
            generator = torch.Generator().manual_seed(seed)

        ts       = int(time.time())
        short_id = uuid.uuid4().hex[:12]
        storage_path = inp.get("storage_path") or f"flux-runs/{ts}-{short_id}.{fmt}"

        image = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        ).images[0]

        content_type = "image/png" if fmt == "png" else "image/jpeg"
        with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as f:
            tmp_path = f.name

        try:
            image.save(tmp_path, format=fmt.upper())
            image_url = upload_to_supabase(tmp_path, storage_path, content_type)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return {
            "image_url":    image_url,
            "format":       fmt,
            "storage_path": storage_path,
            "width":        width,
            "height":       height,
            "seed":         seed,
        }

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

runpod.serverless.start({"handler": handler})
