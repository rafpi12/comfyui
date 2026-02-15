import os, json, aria2p, subprocess, time, uvicorn, shutil, psutil
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

app = FastAPI()

# --- CONFIGURATION SÉCURISÉE ---
BASE_MODELS_PATH = "/workspace/ComfyUI/models"
# Correction du chemin ici :
CONFIG_PATH = "/workspace/model-manager/models.json"

HF_TOKEN = os.environ.get("HF_TOKEN", "") 

MODEL_CATEGORIES = ["checkpoints", "loras", "vae", "upscale_models", "text_encoders", "unet", "diffusion_models", "controlnet"]
ALLOWED_EXTENSIONS = {'.safetensors', '.pth', '.gguf'}

def get_client():
    try:
        client = aria2p.Client(host="http://127.0.0.1", port=6800, secret="")
        api = aria2p.API(client)
        return api
    except: return None

@app.get("/", response_class=HTMLResponse)
async def index(): 
    if os.path.exists("index.html"):
        return open("index.html").read()
    return "Fichier index.html introuvable."

@app.get("/config")
async def get_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f: 
            return json.load(f)
    return {}

@app.post("/save-config")
async def save_config(request: Request):
    data = await request.json()
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    return {"status": "ok"}

@app.get("/disk-state")
async def disk_state():
    state = {}
    for cat in MODEL_CATEGORIES:
        p = os.path.join(BASE_MODELS_PATH, cat)
        if os.path.exists(p):
            state[cat] = [f for f in os.listdir(p) if any(f.endswith(ext) for ext in ALLOWED_EXTENSIONS)]
        else:
            state[cat] = []
    return state

@app.post("/download")
async def download(request: Request):
    data = await request.json()
    url = data.get("url")
    target_dir = data.get("path")
    filename = data.get("filename")

    if not target_dir.startswith("/"):
        target_dir = os.path.join(BASE_MODELS_PATH, target_dir)
    
    os.makedirs(target_dir, exist_ok=True)
    
    client = get_client()
    if not client: return {"status": "error", "message": "Aria2 non connecté"}

    options = {"dir": target_dir, "out": filename}
    if "huggingface.co" in url.lower() and HF_TOKEN:
        options["header"] = f"Authorization: Bearer {HF_TOKEN}"
    
    try:
        client.add(url, options=options)
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/progress")
async def progress():
    client = get_client()
    if not client: return []
    try:
        downloads = client.get_downloads()
        return [{
            "name": d.name,
            "status": d.status,
            "progress": d.progress,
            "speed": d.download_speed_string,
            "eta": d.eta_string
        } for d in downloads]
    except: return []

@app.post("/purge")
async def purge():
    client = get_client()
    if client: client.purge()
    return {"status": "ok"}

@app.delete("/delete")
async def delete(cat: str, file: str):
    target_dir = cat if cat.startswith("/") else os.path.join(BASE_MODELS_PATH, cat)
    p = os.path.join(target_dir, file)
    if os.path.exists(p): os.remove(p)
    return {"status": "ok"}

if __name__ == "__main__":
    # Nettoyage et lancement d'Aria2
    subprocess.run(["pkill", "-9", "aria2c"])
    subprocess.Popen(["aria2c", "--enable-rpc", "--rpc-listen-all=true", "--rpc-allow-origin-all=true", "-D"])
    time.sleep(1)
    # Lancement Uvicorn sur 0.0.0.0 pour RunPod
    uvicorn.run(app, host="0.0.0.0", port=8080)
