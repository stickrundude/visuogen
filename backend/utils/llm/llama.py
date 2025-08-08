from llama_cpp import Llama
from pathlib import Path
import json
import re

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "llama" / "llama-3.1-8b-instruct.Q5_K_M.gguf"
llm = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=4096,
    chat_format="llama-3"
)

def generate_with_messages(system: str, user: str, max_tokens: int = 1500) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.8,
        top_p=0.9,
    )
    return response["choices"][0]["message"]["content"]

def extract_character_traits(prompt: str) -> str:
    system = (
        "You are a summarizer that extracts key visual traits from creative prompts.\n"
        "Output a one-line description of the main character that includes their name (if known), species or role, and key appearance traits.\n"
        "Do not include background details or scene information.\n"
        "Only respond with the description. Do not include preambles or labels."
    )
    output = generate_with_messages(system, f"Prompt: {prompt}")
    lines = output.strip().splitlines()

    for line in lines:
        if line.count(",") >= 2 and len(line.split()) > 6 and not re.match(r"(?i)^\s*(name|species|appearance)", line):
            return line.strip().rstrip(".") + "."
    for line in sorted(lines, key=len, reverse=True):
        if len(line.strip()) > 30 and not re.match(r"(?i)^\s*(name|species|appearance|sure)", line):
            return line.strip().rstrip(".") + "."

    return prompt.strip()

def llama_chat_story(prompt: str, beats: int = 10) -> str:
    character_summary = extract_character_traits(prompt)
    system = (
        f"You are a visual storyteller that generates {beats} scenes featuring a consistent character.\n"
        f"Each scene should reflect the character's appearance and personality.\n"
        f"Use vivid imagery and avoid repeating phrases.\n"
        f"Only output the {beats} scenes, no greetings, no explanations."
    )
    user = (
        f"Story premise: {prompt}\n"
        f"Main character: {character_summary}\n\n"
        f"Write {beats} vivid, numbered scenes showing the character across different situations but still maintaining a story flow."
    )
    return generate_with_messages(system, user, max_tokens=beats * 80)

def llama_chat_diagram(user_prompt: str) -> str:
    system = "You are a layout assistant that only responds with a valid JSON object describing a software architecture diagram."
    raw = generate_with_messages(system, user_prompt)

    # Remove markdown backticks if present
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.removeprefix("```json").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").strip()
    if cleaned.endswith("```"):
        cleaned = cleaned.removesuffix("```").strip()

    return cleaned
def llama_generate_layout_json(user_prompt: str) -> str:
    import json
    import re

    system = (
    "You are a diagram layout assistant. Based on a system description, respond with a **valid raw JSON object**.\n"
    "Return ONLY the object, with no explanations, no markdown, no code blocks.\n"
    "Structure: {\n"
    "  \"nodes\": [{\"name\": ..., \"type\": ...}],\n"
    "  \"connections\": [{\"from\": ..., \"to\": ...}]\n"
    "}\n"
    "Use clear types like 'user', 'gateway', 'web_server', 'database'.\n"
    "Do NOT add anything else. No markdown. No text. Just JSON."
    )

 
    raw = generate_with_messages(system, user_prompt)

    # Remove markdown artifacts
    cleaned = re.sub(r"^```(?:json)?", "", raw.strip(), flags=re.MULTILINE)
    cleaned = cleaned.replace("```", "").strip()

    try:
        parsed = json.loads(cleaned)

        # normalize if "components" is used instead of "nodes"
        if "components" in parsed and "nodes" not in parsed:
            parsed["nodes"] = parsed.pop("components")

        nodes = parsed.get("nodes", [])
        connections = parsed.get("connections", [])

        if not nodes or not connections:
            raise ValueError("Missing 'nodes' or 'connections' in response.")

        for node in nodes:
            node.pop("x", None)
            node.pop("y", None)

        canvas_width, canvas_height = 1920, 1080
        margin_x = 150
        spacing_x = (canvas_width - 2 * margin_x) // max(1, len(nodes) - 1)
        center_y = canvas_height // 2

        for i, node in enumerate(nodes):
            node["x"] = margin_x + spacing_x * i
            node["y"] = center_y

        return json.dumps({
        "nodes": parsed.get("nodes", parsed.get("components", [])),
        "connections": parsed.get("connections", []),
        "prompt": user_prompt  
        })

    except json.JSONDecodeError as e:
        print("[ERROR] Failed to parse LLaMA JSON output:")
        print(cleaned)
        raise ValueError(f"Invalid JSON: {e}")

def llama_generate_icon_labels(user_prompt: str) -> list[str]:
    system = (
        "You're an assistant that extracts the types of components in a technical system prompt.\n"
        "Respond ONLY with a JSON array of lowercase component types in snake_case (e.g., 'app_server').\n"
        "Do NOT include names or connections, explanations, backticks, or markdown. Just raw JSON array."
    )
    raw = generate_with_messages(system, user_prompt)
    cleaned = "\n".join(
        line for line in raw.strip().splitlines()
        if not line.strip().startswith("```")
    ).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [t.lower().replace(" ", "_") for t in parsed if isinstance(t, str)]
    except Exception:
        pass

    raise ValueError(f"Failed to extract icon labels from LLaMA:\n{raw}")