import sys
import os
import json
import gc
from pathlib import Path
from datetime import datetime
from PIL import Image
import torch
import re

from transformers import CLIPTokenizer
from safetensors.torch import load_file

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "kandinsky3-diffusers" / "src"))

from ...utils.llm.llama import llama_chat_story, extract_character_traits
from kandinsky3 import get_T2I_pipeline

DEVICE = torch.device("cpu")
DTYPE = torch.float32
STYLE_DESC = "storybook illustration, soft colors, dreamy atmosphere, subtle line art"

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

def truncate_prompt(prompt: str, max_tokens: int = 77) -> str:
    tokens = tokenizer.tokenize(prompt)
    return prompt if len(tokens) <= max_tokens else tokenizer.convert_tokens_to_string(tokens[:max_tokens])


def make_logger(log_fn):
    return log_fn if log_fn else print

def generate_story_images_kandinsky(
    prompt: str,
    output_dir: Path,
    seed: int = 42,
    beats: int = 8,
    steps: int = 30,
    guidance_scale: float = 8.0,
    cancel_event=None,
    log_fn=None
):
    log = make_logger(log_fn)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    if cancel_event and cancel_event.is_set():
        raise RuntimeError("Cancelled before generation started.")

    character_summary = extract_character_traits(prompt)
    story_raw = llama_chat_story(prompt, beats)

    raw_lines = story_raw.strip().splitlines()
    scenes = []
    for line in raw_lines:
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Cancelled during scene parsing.")

        clean = re.sub(r"^\d+[\).\s]+", "", line).strip()
        if clean:
            if not clean.endswith('.'):
                clean += '.'
            scenes.append(clean)

    pipe = get_T2I_pipeline(
        device_map={"unet": DEVICE, "text_encoder": DEVICE, "movq": DEVICE},
        dtype_map={"unet": DTYPE, "text_encoder": DTYPE, "movq": DTYPE},
    )

    generated_paths = []

    for i, scene in enumerate(scenes, 1):
        if cancel_event and cancel_event.is_set():
            raise RuntimeError(f"Cancelled during generation of scene {i}.")

        final_prompt = (
            f"Style: {STYLE_DESC}. {scene} "
            f"Depict in dreamy, soft illustration style. Character appears consistent: {character_summary}."
        )
        log(f"[T2I] Generating scene {i} with prompt:\n{final_prompt}")

        try:
            image = pipe(
                text=final_prompt,
                guidance_scale=guidance_scale,
                steps=steps,
                images_num=1
            )[0].convert("RGB")

            img_path = output_dir / f"story_{i:02d}.png"
            image.save(img_path)
            generated_paths.append(str(img_path))
            log(f"[✔] Saved scene {i}: {img_path}")
        except Exception as e:
            log(f"[✖] Error in scene {i}: {e}")
        finally:
            if 'image' in locals():
                del image
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if cancel_event and cancel_event.is_set():
        raise RuntimeError("Cancelled after generation, before saving log.")

    log = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "character_summary": character_summary,
        "seed": seed,
        "guidance_scale": guidance_scale,
        "steps": steps,
        "scenes": [{"scene": i+1, "text": s} for i, s in enumerate(scenes)]
    }
    (output_dir / "story_log.json").write_text(json.dumps(log, indent=2))
    log(f"[INFO] Story log saved to {output_dir / 'story_log.json'}")

    return generated_paths