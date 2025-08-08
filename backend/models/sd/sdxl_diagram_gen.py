import json
import torch
from pathlib import Path
from datetime import datetime
from PIL import Image
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline
)
from ...utils.llm.llama import llama_chat_diagram, llama_generate_icon_labels
from threading import Event

DEVICE = torch.device("cpu")
DTYPE = torch.float32

def diagram_flow(
    user_prompt: str,
    output_path: Path,
    seed: int = 42,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    base_ratio: float = 0.8,
    cancel_event: Event = None,
    log_fn=None
) -> str:
    def log(msg: str):
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    def check_cancel():
        if cancel_event and cancel_event.is_set():
            log("[CANCEL] Diagram generation aborted.")
            raise RuntimeError("Cancelled")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    check_cancel()
    log("[DEBUG] Calling LLaMA to generate structured layout JSON...")
    layout_raw = llama_chat_diagram(user_prompt)

    if not layout_raw or not layout_raw.strip():
        raise ValueError("LLaMA returned an empty response.")

    try:
        layout = json.loads(layout_raw) if isinstance(layout_raw, str) else layout_raw
    except Exception as e:
        log("[ERROR] Failed to parse LLaMA JSON output:")
        log(layout_raw)
        raise ValueError(f"Invalid JSON from llama_chat_diagram: {e}")

    components = layout.get("components", [])
    connections = layout.get("connections", [])
    labels = sorted(set(comp["type"].lower().replace(" ", "_") for comp in components))
    if not labels:
        labels = llama_generate_icon_labels(user_prompt)

    sdxl_prompt = user_prompt
    log(f"[DEBUG] Extracted Labels: {labels}")

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    check_cancel()
    log("[DEBUG] Loading SDXL base pipeline...")
    base_pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=DTYPE,
        variant="fp16" if DTYPE == torch.float16 else None,
        use_safetensors=True
    ).to(DEVICE)

    check_cancel()
    log("[DEBUG] Loading SDXL refiner pipeline...")
    refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=DTYPE,
        variant="fp16" if DTYPE == torch.float16 else None,
        use_safetensors=True
    ).to(DEVICE)

    if not isinstance(base_ratio, float):
        raise TypeError(f"Expected 'base_ratio' to be float, got {type(base_ratio).__name__}")

    base_steps = int(num_inference_steps * base_ratio)
    refiner_steps = num_inference_steps - base_steps
    log(f"[INFO] Base steps: {base_steps}, Refiner steps: {refiner_steps}")

    check_cancel()
    log("[DEBUG] Generating base image...")
    base_output = base_pipe(
        prompt=sdxl_prompt,
        num_inference_steps=base_steps,
        guidance_scale=guidance_scale,
        generator=generator
    )
    base_image = base_output.images[0]

    check_cancel()
    log("[DEBUG] Refining image with refiner...")
    refined_output = refiner_pipe(
        prompt=sdxl_prompt,
        image=base_image,
        num_inference_steps=refiner_steps,
        guidance_scale=guidance_scale,
        generator=generator
    )
    refined_image = refined_output.images[0]

    check_cancel()
    refined_image.convert("RGB").save(output_path)
    log(f"[INFO] Final refined SDXL diagram saved at {output_path}")

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "user_prompt": user_prompt,
        "canvas_size": (1024, 1024),
        "icon_labels": labels,
        "output_path": str(output_path),
        "components": components,
        "connections": connections,
        "seed": seed
    }
    (output_path.parent / "diagram_log.json").write_text(json.dumps(log_data, indent=2))

    return str(output_path)