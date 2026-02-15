import os, json, aria2p, subprocess, time, uvicorn, shutil, psutil, requests, base64
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse

app = FastAPI()

# --- CONFIGURATION ---
BASE_MODELS_PATH = "/workspace/ComfyUI/models"
CONFIG_PATH = "/workspace/model-manager/models.json"

HF_TOKEN = os.environ.get("HF_TOKEN", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
REPO_OWNER = "rafpi12"
REPO_NAME = "comfyui"
GITHUB_FILE_PATH = "model-manager/models.json"

MODEL_CATEGORIES = ["checkpoints", "loras", "vae", "upscale_models", "text_encoders", "unet", "diffusion_models", "controlnet"]
ALLOWED_EXTENSIONS = {'.safetensors', '.pth', '.gguf'}

def sync_to_github():
    if not GITHUB_TOKEN: return False
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{GITHUB_FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    try:
        r = requests.get(url, headers=headers)
        sha = r.json().get('sha') if r.status_code == 200 else None
        with open(CONFIG_PATH, "rb") as f:
            content = base64.b64encode(f.read()).decode()
        payload = {"message": "Update models.json via Manager", "content": content, "sha": sha}
        requests.put(url, headers=headers, json=payload)
        return True
    except: return False

@app.get("/fetch-civitai-name")
async def fetch_civitai_name(url: str):
    """Tente de récupérer le nom du fichier depuis l'API Civitai"""
    try:
        # Extraire l'ID du modèle depuis l'URL
        # Exemple: https://civitai.com/api/download/models/2687961
        model_id = url.split('/')[-1].split('?')[0]
        r = requests.head(url, allow_redirects=True)
        # On regarde dans les headers de réponse (Content-Disposition)
        cd = r.headers.get('content-disposition')
        if cd and 'filename=' in cd:
            fname = cd.split('filename=')[1].strip('"')
            return {"filename": fname}
        return {"filename": f"model_{model_id}.safetensors"}
    except:
        return {"filename": ""}

@app.get("/", response_class=HTMLResponse)
async def index(): 
    if os.path.exists("index.html"): return open("index.html").read()
    return "Fichier index.html introuvable."

@app.get("/config")
async def get_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f: return json.load(f)
    return {}

@app.post("/save-config")
async def save_config(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    background_tasks.add_task(sync_to_github)
    return {"status": "ok", "github_sync": "started"}

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
                        rel_path = os.path.relpath(os.path.join(root, f), base_path)
                        files.append(rel_path.replace("\\", "/"))
        res[cat] = files
    return res

@app.get("/check-file")
async def check_file(category: str, filename: str):
    path = os.path.join(BASE_MODELS_PATH, category, filename)
    return {"exists": os.path.exists(path), "path": path}

@app.post("/download")
async def download(request: Request):
    data = await request.json()
    url, category, filename = data.get("url"), data.get("path"), data.get("filename")
    target_dir = os.path.join(BASE_MODELS_PATH, category)
    os.makedirs(target_dir, exist_ok=True)
    client = get_client()
    if not client: return {"status": "error", "message": "Aria2 non connecté"}
    options = {"dir": target_dir, "out": filename}
    if "huggingface.co" in url.lower() and HF_TOKEN:
        options["header"] = f"Authorization: Bearer {HF_TOKEN}"
    try:
        client.add(url, options=options)
        return {"status": "ok"}
    except Exception as e: return {"status": "error", "message": str(e)}

@app.get("/progress")
async def progress():
    client = get_client()
    if not client: return []
    try:
        downloads = client.get_downloads()
        return [{"name": d.name, "status": d.status, "progress": d.progress, "speed": d.download_speed_string, "eta": d.eta_string} for d in downloads]
    except: return []

@app.delete("/delete")
async def delete(cat: str, file: str):
    p = os.path.join(BASE_MODELS_PATH, cat, file)
    if os.path.exists(p): os.remove(p)
    return {"status": "ok"}

if __name__ == "__main__":
    subprocess.run(["pkill", "-9", "aria2c"])
    subprocess.Popen(["aria2c", "--enable-rpc", "--rpc-listen-all=true", "--rpc-allow-origin-all=true", "-D"])
    time.sleep(1)
    uvicorn.run(app, host="0.0.0.0", port=8080)
