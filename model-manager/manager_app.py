import os, json, aria2p, subprocess, time, uvicorn, shutil, psutil, requests, base64, re
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse

app = FastAPI()

# --- CONFIGURATION ---
BASE_MODELS_PATH = "/workspace/ComfyUI/models"
CONFIG_PATH = "/workspace/model-manager/models.json"

HF_TOKEN = os.environ.get("HF_TOKEN", "")
CIVITAI_TOKEN = os.environ.get("CIVITAI_TOKEN", "") 
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

def sync_to_github():
    if not GITHUB_TOKEN: return False
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{GITHUB_FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    try:
        r = requests.get(url, headers=headers)
        sha = r.json().get('sha') if r.status_code == 200 else None
        with open(CONFIG_PATH, "rb") as f:
            content = base64.b64encode(f.read()).decode()
        payload = {"message": "Update models.json via Model Manager Pro", "content": content, "sha": sha}
        requests.put(url, headers=headers, json=payload)
        return True
    except: return False

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
    try:
        if "civitai.com" in url:
            model_id = url.split('/models/')[1].split('?')[0]
            api_url = f"https://civitai.com/api/v1/model-versions/{model_id}"
            headers = {"Authorization": f"Bearer {CIVITAI_TOKEN}"} if CIVITAI_TOKEN else {}
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
async def save_config(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    background_tasks.add_task(sync_to_github)
    return {"status": "ok", "github_sync": "started"}

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

@app.get("/check-file")
async def check_file(category: str, filename: str):
    path = os.path.join(BASE_MODELS_PATH, category, filename)
    return {"exists": os.path.exists(path), "path": path}

@app.post("/download")
async def download(request: Request):
    data = await request.json()
    url, category, filename = data.get("url"), data.get("path"), data.get("filename")
    clean_cat = category.replace(BASE_MODELS_PATH, "").lstrip("/")
    target_dir = os.path.join(BASE_MODELS_PATH, clean_cat)
    os.makedirs(target_dir, exist_ok=True)
    
    client = get_client()
    if not client: return {"status": "error", "message": "Aria2 non connecté"}

    is_civitai = "civitai.com" in url.lower()
    
    # Configuration optimisée pour éviter l'Error 22 sur Civitai
    options = {
        "dir": target_dir, 
        "out": filename, 
        "continue": "true",
        "max-connection-per-server": "1" if is_civitai else "16",
        "split": "1" if is_civitai else "16",
        "header": ["User-Agent: Wget/1.21.2", "Accept: */*"]
    }

    final_url = url
    if "huggingface.co" in url.lower():
        options["header"].append(f"Authorization: Bearer {HF_TOKEN}")
    elif is_civitai:
        sep = "&" if "?" in url else "?"
        final_url = f"{url}{sep}token={CIVITAI_TOKEN}"

    try:
        dest = os.path.join(target_dir, filename)
        if os.path.exists(dest + ".aria2"): os.remove(dest + ".aria2")
        client.add(final_url, options=options)
        return {"status": "ok"}
    except Exception as e: return {"status": "error", "message": str(e)}

@app.get("/progress")
async def progress():
    client = get_client()
    if not client: return []
    try:
        downloads = client.get_downloads()
        res = []
        for d in downloads:
            # Sécurité pour le nom du fichier au début du téléchargement
            display_name = d.name if (d.name and not d.name.startswith('http')) else "Initialisation..."
            res.append({
                "name": display_name, 
                "status": f"{d.status} (Code: {d.error_code})" if d.status == 'error' else d.status, 
                "progress": d.progress if d.progress is not None else 0, 
                "speed": d.download_speed_string, 
                "eta": d.eta_string
            })
        return res
    except: return []

@app.delete("/delete")
async def delete(cat: str, file: str):
    clean_cat = cat.replace(BASE_MODELS_PATH, "").lstrip("/")
    p = os.path.join(BASE_MODELS_PATH, clean_cat, file)
    if os.path.exists(p): os.remove(p)
    return {"status": "ok"}

@app.post("/purge")
async def purge():
    client = get_client()
    if client: client.purge()
    return {"status": "ok"}

if __name__ == "__main__":
    subprocess.run(["pkill", "-9", "aria2c"])
    subprocess.Popen([
        "aria2c", 
        "--enable-rpc", 
        "--rpc-listen-all=true", 
        "--rpc-allow-origin-all=true", 
        "--max-concurrent-downloads=3",
        "--follow-torrent=mem",
        "-D"
    ])
    time.sleep(1)
    uvicorn.run(app, host="0.0.0.0", port=8080)
