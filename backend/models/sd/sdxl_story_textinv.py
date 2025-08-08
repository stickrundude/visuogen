import sys
import os
import uuid
import json
import runpy
from pathlib import Path
from datetime import datetime
from typing import Optional
from threading import Event

import torch
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import load_file as load_safetensors
from transformers import CLIPTokenizer

from ...utils.llm.llama import llama_chat_story, extract_character_traits


def make_logger(log_fn):
    return log_fn if log_fn else print


if not hasattr(torch, "get_default_device"):
    def get_default_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    torch.get_default_device = get_default_device

DEVICE = torch.get_default_device()
IS_MPS = DEVICE.type == "mps"
DTYPE = torch.float16 if IS_MPS else torch.float32
print(f"[INFO] Using device: {DEVICE}, dtype: {DTYPE}")

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

def truncate_prompt(prompt: str, max_tokens: int = 77) -> str:
    tokens = tokenizer.tokenize(prompt)
    return prompt if len(tokens) <= max_tokens else tokenizer.convert_tokens_to_string(tokens[:max_tokens])

# === Token Registry ===
def load_token_registry(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}

def save_token_registry(registry: dict, path: Path):
    path.write_text(json.dumps(registry, indent=2))

def get_or_create_token(description: str, path: Path) -> str:
    registry = load_token_registry(path)
    for tok, meta in registry.items():
        if meta["description"] == description:
            print(f"[INFO] Reusing token: {tok}")
            return tok
    token = f"<char_{uuid.uuid4().hex[:8]}>"
    registry[token] = {"description": description}
    print(f"[INFO] Created new token: {token}")
    save_token_registry(registry, path)
    return token

def load_textinv_embedding(pipe: StableDiffusionXLPipeline, embedding_dir: Path, token: str):
    e1 = embedding_dir / "learned_embeds.safetensors"
    e2 = embedding_dir / "learned_embeds_2.safetensors"
    embeds1 = load_safetensors(e1)
    embeds2 = load_safetensors(e2)

    if token not in pipe.tokenizer.get_vocab():
        pipe.tokenizer.add_tokens([token])
        pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
        pipe.text_encoder_2.resize_token_embeddings(len(pipe.tokenizer))

    tid = pipe.tokenizer.convert_tokens_to_ids(token)
    with torch.no_grad():
        pipe.text_encoder.get_input_embeddings().weight[tid] = embeds1[token]
        pipe.text_encoder_2.get_input_embeddings().weight[tid] = embeds2[token]

    return pipe

def storybot_flow(user_prompt: str, base_dir: Path, seed: int, cancel_event: Optional[Event] = None, log_fn=None) -> list[str]:
    log = make_logger(log_fn)
    
    character_summary = extract_character_traits(user_prompt)
    registry_path = base_dir / "character_token_registry.json"
    token = get_or_create_token(character_summary, registry_path)

    char_dir = base_dir / "textinv_train" / "char"
    textinv_dir = base_dir / "textinv_embedding"
    story_dir = base_dir / "story"
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    if cancel_event and cancel_event.is_set():
        raise RuntimeError("Cancelled before generation")

    if not (textinv_dir / "learned_embeds.safetensors").exists():
        log(f"[INFO] No embeddings found, starting training...")
        generate_character_variants(character_summary, char_dir, generator, count=100, cancel_event=cancel_event, log_fn=log)
        train_textinv(char_dir, textinv_dir, token, max_steps=800, cancel_event=cancel_event, log_fn=log)
    else:
        log("[INFO] Using cached text inversion embeddings.")

    return generate_story_images(
        character_summary,
        token,
        textinv_dir,
        story_dir,
        generator,
        beats=10,
        cancel_event=cancel_event,
        log_fn=log
    )

def generate_story_images(
    description: str,
    token: str,
    embedding_dir: Path,
    output_dir: Path,
    generator: torch.Generator,
    beats: int = 10,
    cancel_event: Optional[Event] = None,
    log_fn=None
) -> list[str]:
    log = make_logger(log_fn)

    if cancel_event and cancel_event.is_set():
        raise RuntimeError("Cancelled before story generation")

    character_summary = extract_character_traits(description)
    raw = llama_chat_story(character_summary, beats)

    lines = []
    for l in raw.splitlines():
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Cancelled during prompt processing")
        l = l.strip()
        if not l:
            continue
        if len(l) > 2 and l[:2].isdigit() and l[1] in [".", ")"]:
            l = l[2:].strip()
        lines.append(l)

    if not lines:
        raise ValueError("[ERROR] No valid story lines found.")
    if len(lines) < beats:
        lines += [lines[-1]] * (beats - len(lines))
    else:
        lines = lines[:beats]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=DTYPE,
        variant="fp16" if IS_MPS else None,
        use_safetensors=True,
    ).to(DEVICE)
    pipe = load_textinv_embedding(pipe, embedding_dir, token)

    for i, scene in enumerate(lines, start=1):
        if cancel_event and cancel_event.is_set():
            log(f"[CANCEL] Aborting story at image {i}")
            raise RuntimeError("Cancelled during story generation")

        prompt = truncate_prompt(f"{token}, a {character_summary}, in a scene where {scene}")
        log(f"[DEBUG] Generating image {i}/{beats}: {prompt}")
        img = pipe(
            prompt=prompt,
            num_inference_steps=50,
            guidance_scale=10.0,
            generator=generator
        ).images[0]
        fp = output_dir / f"story_{i:02d}.png"
        img.save(fp)
        out_paths.append(str(fp))

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "prompt": description,
        "character_summary": character_summary,
        "character_token": token,
        "seed": generator.initial_seed(),
        "guidance_scale": 10.0,
        "steps": 50,
        "scenes": [{"scene": i+1, "text": line} for i, line in enumerate(lines)]
    }
    (output_dir / "story_log.json").write_text(json.dumps(log_data, indent=2))
    log(f"[INFO] Story log saved at {output_dir / 'story_log.json'}")

    return out_paths

def generate_character_variants(
    description: str,
    output_dir: Path,
    generator: torch.Generator,
    count: int = 10,
    cancel_event: Optional[Event] = None,
    log_fn=None
):
    log = make_logger(log_fn)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=DTYPE,
        variant="fp16" if IS_MPS else None,
        use_safetensors=True,
    ).to(DEVICE)

    for i in range(count):
        if cancel_event and cancel_event.is_set():
            log(f"[CANCEL] Aborting character variant generation at {i}")
            raise RuntimeError("Cancelled during variant generation")

        p = f"{description}, solo character, cinematic lighting, plain background"
        img = pipe(prompt=p, num_inference_steps=40, guidance_scale=7.5, generator=generator).images[0]
        img.save(output_dir / f"character_{i:02d}.png")

def train_textinv(
    instance_data_dir: Path,
    output_dir: Path,
    token: str,
    max_steps: int,
    cancel_event: Optional[Event] = None,
    log_fn=None
):
    log = make_logger(log_fn)
    if cancel_event and cancel_event.is_set():
        log("[CANCEL] Skipping training due to cancellation request")
        raise RuntimeError("Cancelled before training")
    
    log(f"[INFO] Training TI token: {token}")
    os.environ.pop("USE_MPS_DEVICE", None)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.argv = [
        "textual_inversion_sdxl.py",
        "--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0",
        f"--train_data_dir={instance_data_dir}",
        "--learnable_property=object",
        f"--placeholder_token={token}",
        "--initializer_token=person",
        "--resolution=512",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=1",
        f"--max_train_steps={max_steps}",
        f"--output_dir={output_dir}",
    ]
    torch.mps.empty_cache()
    runpy.run_path("diffusers/examples/textual_inversion/textual_inversion_sdxl.py", run_name="__main__")