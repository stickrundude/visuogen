# VisuoGen

**VisuoGen** is a prompt-driven tool for generating **visual stories** and **structured diagrams** using generative AI models. It combines a FastAPI backend with a React frontend and uses LLaMA 3, Stable Diffusion XL and Kandinsky 3 to turn simple text prompts into coherent visuals.

---

##  Features

- Generate illustrated **story scenes** from a single prompt  
- Create structured **diagrams** using prompt-to-layout conversion  
- **Model support**: Stable Diffusion XL and Kandinsky 3  
- **Prompt decomposition** with local LLaMA 3.1–8B  
- **Textual Inversion** for consistent character generation with SDXL 
- **Live log streaming** during generation  
- **Speech-to-text** prompt input (in browser)

---

##  Tech Stack

### Backend (FastAPI)
- `fastapi`, `uvicorn`
- `torch`, `diffusers`, `transformers`
- `safetensors`, `pillow`
- `llama-cpp-python` (for local LLaMA model)

### Frontend (React)
- React with `react-router-dom`
- Plain CSS and inline styling
- Voice input via `webkitSpeechRecognition`
- Fetch-based communication with backend

---

##  Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/stickrundude/visuogen.git
cd visuogen
```

### 2. Backend setup (Python)

Create a virtual environment and install dependencies:

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Make sure you have the SDXL weights, Kandinsky pipeline, and LLaMA model in place.

Run the backend:

```bash
uvicorn main:app --reload
```

### 3. Frontend setup (React)

```bash
cd frontend/visuogen-ui
npm install
npm run dev
```

The app will run at [http://localhost:3000](http://localhost:3000)  
Backend is expected at [http://localhost:8000](http://localhost:8000)

---

##  Project Structure

```
visuogen/
├── backend/
│   ├── main.py
│   ├── models/
│   ├── utils/llm/llama.py
│   └── static/       ← generated images + logs
├── frontend/visuogen-ui/
│   ├── src/components/
│   ├── src/pages/
│   ├── App.js
│   ├── index.js 
├── requirements.txt
└── README.md
```

---

##  Output Examples

- `story_log.json`: contains prompt, character, scene texts
- `diagram_log.json`: contains layout JSON, prompt, icon labels
- `story_01.png`, `diagram.png`: visual outputs per job

---

##  Author

Made by **Kshitij Deshmukh** as part of a master thesis project at SRH University Heidelberg.

---

##  License

This project is for educational and non-commercial research use. All models used (LLaMA, SDXL, Kandinsky) follow their respective licenses.
