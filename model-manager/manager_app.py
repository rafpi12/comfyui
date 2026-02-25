import os, json, aria2p, subprocess, time, uvicorn, shutil, psutil, requests, base64, re
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse
from urllib.parse import urlparse, unquote

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

@app.post("/download")
async def download(request: Request):
    data = await request.json()
    url, category, filename = data.get("url"), data.get("path"), data.get("filename")
    clean_cat = category.replace(BASE_MODELS_PATH, "").lstrip("/")
    target_dir = os.path.join(BASE_MODELS_PATH, clean_cat)
    os.makedirs(target_dir, exist_ok=True)

    client = get_client()
    if not client: return {"status": "error", "message": "Aria2 non connecté"}

    # Nettoyage préventif
    dest = os.path.join(target_dir, filename)
    if os.path.exists(dest + ".aria2"): os.remove(dest + ".aria2")

    is_civitai = "civitai.com" in url.lower()
    is_hf = "huggingface.co" in url.lower()

    headers = [
        "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ]

    final_url = url

    if is_civitai:
        # Résoudre la redirection 307 côté Python AVANT de passer à Aria2.
        # Civitai renvoie une URL Backblaze B2 signée (token temporaire dans la query).
        # Si Aria2 suit lui-même la redirection, il transmet le header Authorization
        # sur b2.civitai.com, ce qui invalide le token signé → Error 22.
        # En résolvant ici, Aria2 appelle directement l'URL signée sans header d'auth.
        try:
            resolve_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "Authorization": f"Bearer {CIVITAI_TOKEN}"
            }
            r = requests.get(url, headers=resolve_headers, allow_redirects=False, timeout=10)
            if r.status_code in (301, 302, 307, 308) and "location" in r.headers:
                final_url = r.headers["location"]
            elif r.status_code != 200:
                # Fallback : token en query param
                sep = "&" if "?" in url else "?"
                final_url = f"{url}{sep}token={CIVITAI_TOKEN}"
        except Exception as e:
            return {"status": "error", "message": f"Impossible de résoudre l'URL Civitai : {e}"}

        # L'URL B2 signée est publique — pas de header Authorization à envoyer
        # On peut utiliser toutes les connexions pour la vitesse maximale
        connections = "16"
        split = "16"

    elif is_hf:
        if HF_TOKEN:
            headers.append(f"Authorization: Bearer {HF_TOKEN}")
        connections = "16"
        split = "16"
    else:
        connections = "16"
        split = "16"

    options = {
        "dir": target_dir,
        "out": filename,
        "continue": "true",
        "allow-overwrite": "true",
        "auto-file-renaming": "false",
        "max-connection-per-server": connections,
        "split": split,
        "min-split-size": "10M",
        "header": headers,
        "check-certificate": "false",
        "max-tries": "5",
        "retry-wait": "3",
    }

    try:
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
        res = []
        for d in downloads:
            # Pendant l'init, d.name peut contenir l'URL — on extrait le vrai nom
            raw_name = d.name or ""
            if raw_name.startswith("http"):
                name = filename_fallback(d)
            else:
                name = raw_name

            # Fallback : extraire depuis l'URI Aria2 si toujours vide
            if not name or name == "Initialisation...":
                try:
                    parsed = urlparse(d.files[0].uris[0]["uri"] if d.files else "")
                    name = unquote(parsed.path.split("/")[-1]) or "Initialisation..."
                except:
                    name = "Initialisation..."

            status = d.status
            if status == "error":
                status = f"Erreur (Code {d.error_code})"

            # download_speed_string et eta_string sont des méthodes — appel avec ()
            try:
                speed_bps = d.download_speed or 0
                if speed_bps >= 1_000_000:
                    speed_str = f"{speed_bps / 1_000_000:.1f} Mo/s"
                elif speed_bps >= 1000:
                    speed_str = f"{speed_bps / 1000:.0f} Ko/s"
                else:
                    speed_str = f"{speed_bps} o/s"
            except:
                speed_str = "0 o/s"

            try:
                eta_secs = int(d.eta.total_seconds()) if hasattr(d.eta, 'total_seconds') else int(d.eta or 0)
                if eta_secs > 3600:
                    eta_str = f"{eta_secs // 3600}h{(eta_secs % 3600) // 60}m"
                elif eta_secs > 60:
                    eta_str = f"{eta_secs // 60}m{eta_secs % 60}s"
                elif eta_secs > 0:
                    eta_str = f"{eta_secs}s"
                else:
                    eta_str = ""
            except:
                eta_str = ""

            res.append({
                "name": name,
                "status": status,
                "progress": round(d.progress, 1),
                "speed": speed_str,
                "eta": eta_str,
                "gid": d.gid,
            })
        return res
    except: return []

def filename_fallback(download):
    try:
        return download.files[0].path.split('/')[-1]
    except:
        return "Initialisation..."

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
    time.sleep(1)
    subprocess.Popen([
        "aria2c", "--enable-rpc", "--rpc-listen-all=true",
        "--rpc-allow-origin-all=true", "--max-concurrent-downloads=3",
        "--follow-torrent=mem", "--rpc-save-upload-metadata=true", "-D"
    ])
    uvicorn.run(app, host="0.0.0.0", port=8080)
