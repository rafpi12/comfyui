import os, json, aria2p, subprocess, time, uvicorn, shutil, psutil, requests, base64, re
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse

app = FastAPI()

# --- CONFIGURATION ---
BASE_MODELS_PATH = "/workspace/ComfyUI/models"
CONFIG_PATH = "/workspace/model-manager/models.json"

# RÉCUPÉRATION SÉCURISÉE DES TOKENS (Variables d'environnement RunPod)
HF_TOKEN = os.environ.get("HF_TOKEN", "")
CIVITAI_TOKEN = os.environ.get("CIVITAI_TOKEN", "") # Récupère la clé du Pod Template
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

REPO_OWNER = "rafpi12"
REPO_NAME = "comfyui"
GITHUB_FILE_PATH = "model-manager/models.json"

ALLOWED_EXTENSIONS = {'.safetensors', '.pth', '.pt', '.gguf', '.bin', '.ckpt', '.yaml'}

def get_client():
    try:
        client = aria2p.Client(host="http://127.0.0.1", port=6800, secret="")
        api = aria2p.API(client)
        return api
    except: return None

# --- ROUTES API ---

@app.get("/list-subfolders")
async def list_subfolders(category: str):
    base = os.path.join(BASE_MODELS_PATH, category)
    subdirs = [""]
    if os.path.exists(base):
        for root, dirs, _ in os.walk(base):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for d in dirs:
                rel = os.path.relpath(os.path.join(root, d), base)
                subdirs.append(rel.replace("\\", "/"))
    return sorted(list(set(subdirs)))

@app.get("/fetch-civitai-name")
async def fetch_civitai_name(url: str):
    if not CIVITAI_TOKEN:
        return {"filename": "ERREUR: Token Civitai manquant"}
    try:
        if "civitai.com" in url:
            model_id_match = re.search(r'/models/(\d+)', url)
            if not model_id_match: model_id_match = re.search(r'models/(\d+)', url)
            if model_id_match:
                model_id = model_id_match.group(1)
                api_url = f"https://civitai.com/api/v1/model-versions/{model_id}"
                headers = {"Authorization": f"Bearer {CIVITAI_TOKEN}"}
                resp = requests.get(api_url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    for file in data.get('files', []):
                        if file.get('primary'): return {"filename": file.get('name')}
                    if data.get('files'): return {"filename": data['files'][0].get('name')}
        return {"filename": ""}
    except: return {"filename": ""}

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
async def save_config(request: Request):
    data = await request.json()
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    return {"status": "ok"}

@app.get("/scan-disk")
async def scan_disk():
    res = {}
    if os.path.exists(BASE_MODELS_PATH):
        for entry in os.scandir(BASE_MODELS_PATH):
            if entry.is_dir():
                cat_name = entry.name
                files = []
                for root, _, filenames in os.walk(entry.path):
                    for f in filenames:
                        if any(f.endswith(ext) for ext in ALLOWED_EXTENSIONS):
                            rel_path = os.path.relpath(os.path.join(root, f), entry.path)
                            files.append(rel_path.replace("\\", "/"))
                res[cat_name] = files
    return res

@app.post("/download")
async def download(request: Request):
    data = await request.json()
    url, category, filename = data.get("url"), data.get("path"), data.get("filename")
    
    if "civitai.com" in url.lower() and not CIVITAI_TOKEN:
        return {"status": "error", "message": "Variable CIVITAI_TOKEN non définie dans le Pod"}

    clean_cat = category.replace(BASE_MODELS_PATH, "").lstrip("/")
    target_dir = os.path.join(BASE_MODELS_PATH, clean_cat)
    os.makedirs(target_dir, exist_ok=True)
    
    client = get_client()
    if not client: return {"status": "error", "message": "Aria2 non connecté"}

    dest = os.path.join(target_dir, filename)
    if os.path.exists(dest + ".aria2"): os.remove(dest + ".aria2")

    is_civitai = "civitai.com" in url.lower()
    
    options = {
        "dir": target_dir, 
        "out": filename, 
        "continue": "true",
        "max-connection-per-server": "4" if is_civitai else "16",
        "split": "4" if is_civitai else "16",
        "min-split-size": "1M",
        "check-certificate": "false",
        "header": ["User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"]
    }

    final_url = url
    if "huggingface.co" in url.lower() and HF_TOKEN:
        options["header"].append(f"Authorization: Bearer {HF_TOKEN}")
    elif is_civitai:
        if "token=" not in url:
            sep = "&" if "?" in url else "?"
            final_url = f"{url}{sep}token={CIVITAI_TOKEN}"

    try:
        print(f"DEBUG: Download start -> {filename}")
        client.add(final_url, options=options)
        return {"status": "ok"}
    except Exception as e: 
        return {"status": "error", "message": str(e)}

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
    clean_cat = cat.replace(BASE_MODELS_PATH, "").lstrip("/")
    p = os.path.join(BASE_MODELS_PATH, clean_cat, file)
    if os.path.exists(p): os.remove(p)
    if os.path.exists(p + ".aria2"): os.remove(p + ".aria2")
    return {"status": "ok"}

@app.post("/purge")
async def purge():
    client = get_client()
    if client: client.purge()
    return {"status": "ok"}

if __name__ == "__main__":
    print(f"STAUTS: Civitai Token trouvé : {'OUI' if CIVITAI_TOKEN else 'NON'}")
    subprocess.run(["pkill", "-9", "aria2c"], stderr=subprocess.DEVNULL)
    time.sleep(1)
    subprocess.Popen([
        "aria2c", "--enable-rpc", "--rpc-listen-all=true", 
        "--rpc-allow-origin-all=true", "--max-concurrent-downloads=3", 
        "--follow-torrent=mem", "--quiet=true", "-D"
    ])
    uvicorn.run(app, host="0.0.0.0", port=8080)
