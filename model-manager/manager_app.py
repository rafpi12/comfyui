import os, json, aria2p, subprocess, time, uvicorn, shutil, psutil
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

app = FastAPI()

# --- CONFIGURATION SÉCURISÉE ---
BASE_MODELS_PATH = "/workspace/ComfyUI/models"
CONFIG_PATH = "/workspace/ComfyUI/custom_nodes/ComfyUI-Personal-Downloader/models.json"

# On récupère le token depuis le système (plus de fuite sur GitHub !)
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
async def index(): return open("index.html").read()

@app.get("/config")
async def get_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f: return json.load(f)
    return {}

@app.post("/save-config")
async def save_config(request: Request):
    data = await request.json()
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    return {"status": "ok"}

@app.get("/check-file")
async def check_file(category: str, filename: str):
    clean_cat = category.replace(BASE_MODELS_PATH, "").lstrip("/")
    p = os.path.join(BASE_MODELS_PATH, clean_cat, filename)
    return {"exists": os.path.exists(p), "path": p}

@app.get("/scan-disk")
async def scan_disk():
    res = {}
    for cat in MODEL_CATEGORIES:
        base_path = os.path.join(BASE_MODELS_PATH, cat)
        files = []
        if os.path.exists(base_path):
            for root, _, filenames in os.walk(base_path):
                for f in filenames:
                    if any(f.endswith(ext) for ext in ALLOWED_EXTENSIONS):
                        rel = os.path.relpath(os.path.join(root, f), base_path)
                        files.append(rel.replace("\\", "/"))
        res[cat] = files
    return res

@app.post("/download")
async def download(url: str, category: str, filename: str):
    client = get_client()
    if not client: return {"status": "error", "message": "Moteur Aria2 hors ligne"}
    
    target_dir = category if category.startswith("/") else os.path.join(BASE_MODELS_PATH, category)
    os.makedirs(target_dir, exist_ok=True)
    
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
            "status": f"{d.status} (Code: {d.error_code})" if d.status == 'error' else d.status,
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
    subprocess.run(["pkill", "-9", "aria2c"])
    subprocess.Popen(["aria2c", "--enable-rpc", "--rpc-listen-all=true", "--rpc-allow-origin-all=true", "-D"])
    time.sleep(1)
    print(f"🚀 Manager Pro lancé (Mode Sécurisé).")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="error")
