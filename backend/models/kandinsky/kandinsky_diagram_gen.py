import json
import gc
import math
import torch
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

from kandinsky3 import get_T2I_pipeline
from backend.utils.llm.llama import llama_generate_layout_json

device = torch.device("cpu")
ICON_SIZE = 256
ICON_CENTER_OFFSET = ICON_SIZE // 2


def make_logger(log_fn):
    return log_fn if log_fn else print

def get_canvas_size(base=(1920, 1080)):
    return base

def generate_icon_for_label(label: str, output_path: Path, device=torch.device("cpu"), seed: int = None, cancel_event=None, log_fn=None):
    log = make_logger(log_fn)

    if cancel_event and cancel_event.is_set():
        log(f"[CANCELLED] Skipping icon generation for {label}")
        raise RuntimeError("Generation cancelled by user.")

    prompt = f"Minimal white flat icon for {label}, centered, plain background, no text"
    log(f"[ICON] Generating icon for: {label}")

    if seed is not None:
        torch.manual_seed(seed)

    device_map = {'unet': device, 'text_encoder': device, 'movq': device}
    dtype_map = {'unet': torch.float32, 'text_encoder': torch.float32, 'movq': torch.float32}

    with torch.no_grad():
        pipe = get_T2I_pipeline(device_map, dtype_map)
        images = pipe(text=prompt, guidance_scale=7.0, steps=30, images_num=1)
        result = images[0].convert("RGBA").resize((ICON_SIZE, ICON_SIZE), Image.LANCZOS)
        result.save(output_path)

        del images
        del pipe
        torch.cuda.empty_cache()
        gc.collect()

def render_diagram_from_layout(layout: dict, background_size: tuple, icons_dir: Path, cancel_event=None, log_fn=None) -> Image:
    log = make_logger(log_fn)

    if cancel_event and cancel_event.is_set():
        log("[CANCELLED] Diagram rendering cancelled")
        raise RuntimeError("Generation cancelled by user.")

    img = Image.new("RGBA", background_size, (240, 240, 240, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except IOError:
        font = ImageFont.load_default()

    node_positions = {}
    for node in layout["nodes"]:
        if cancel_event and cancel_event.is_set():
            log("[CANCELLED] Node placement cancelled")
            raise RuntimeError("Generation cancelled by user.")

        name = node["name"]
        icon_type = node["type"].lower().replace(" ", "_")
        x = int(node["x"])
        y = int(node["y"])
        node_positions[name] = (x, y)

        icon_path = icons_dir / f"{icon_type}.png"
        if icon_path.exists():
            icon_img = Image.open(icon_path).resize((ICON_SIZE, ICON_SIZE))
            img.paste(icon_img, (x, y), mask=icon_img.convert("RGBA"))
        else:
            draw.rectangle([x, y, x + ICON_SIZE, y + ICON_SIZE], fill="gray")

        draw.text((x, y + ICON_SIZE + 10), name, font=font, fill="black")

    for conn in layout.get("connections", []):
        if cancel_event and cancel_event.is_set():
            log("[CANCELLED] Connection drawing cancelled")
            raise RuntimeError("Generation cancelled by user.")

        start = node_positions.get(conn["from"])
        end = node_positions.get(conn["to"])
        if start and end:
            sx = start[0] + ICON_CENTER_OFFSET
            sy = start[1] + ICON_CENTER_OFFSET
            ex = end[0] + ICON_CENTER_OFFSET
            ey = end[1] + ICON_CENTER_OFFSET

            draw.line([sx, sy, ex, ey], fill="black", width=3)

            arrow_size = 18
            angle = math.atan2(ey - sy, ex - sx)
            arrow_x1 = ex - arrow_size * math.cos(angle - math.pi / 6)
            arrow_y1 = ey - arrow_size * math.sin(angle - math.pi / 6)
            arrow_x2 = ex - arrow_size * math.cos(angle + math.pi / 6)
            arrow_y2 = ey - arrow_size * math.sin(angle + math.pi / 6)
            draw.polygon([(ex, ey), (arrow_x1, arrow_y1), (arrow_x2, arrow_y2)], fill="black")

    return img

def generate_diagram_kandinsky3_full(user_prompt: str, output_path: Path, seed: int = None, cancel_event=None, log_fn=None):
    log = make_logger(log_fn)
    log(f"[INFO] Generating structured diagram for prompt: {user_prompt}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if cancel_event and cancel_event.is_set():
        log("[CANCELLED] Aborting before layout")
        raise RuntimeError("Cancelled before layout step")

    layout_raw = llama_generate_layout_json(user_prompt)
    layout = json.loads(layout_raw)

    if not layout.get("nodes") or not layout.get("connections"):
        raise ValueError("Parsed layout is missing 'nodes' or 'connections'")

    canvas_size = get_canvas_size()
    icon_labels = sorted(set(node["type"].lower().replace(" ", "_") for node in layout["nodes"]))
    icons_dir = output_path.parent / "icons"
    icons_dir.mkdir(parents=True, exist_ok=True)

    for label in icon_labels:
        if cancel_event and cancel_event.is_set():
            log("[CANCELLED] During icon creation")
            raise RuntimeError("Generation cancelled during icon creation.")

        icon_path = icons_dir / f"{label}.png"
        if not icon_path.exists():
            log(f"[DEBUG] Icon for '{label}' not found. Generating...")
            generate_icon_for_label(label, icon_path, device=device, seed=seed, cancel_event=cancel_event, log_fn=log)

    Image.new("RGB", canvas_size, (240, 240, 240)).save(output_path.parent / "background.png")

    final_image = render_diagram_from_layout(layout, canvas_size, icons_dir, cancel_event=cancel_event, log_fn=log)
    final_image.save(output_path)
    log(f"[INFO] Saved composed diagram at {output_path}")

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "user_prompt": user_prompt,
        "canvas_size": canvas_size,
        "icon_labels": icon_labels,
        "output_path": str(output_path),
        "nodes": layout.get("nodes", []),
        "connections": layout.get("connections", []),
        "seed": seed
    }
    (output_path.parent / "diagram_log.json").write_text(json.dumps(log_data, indent=2))
    return str(output_path)