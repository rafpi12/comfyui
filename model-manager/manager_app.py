import os, json, aria2p, subprocess, time, uvicorn, shutil, psutil, requests, base64, re
from fastapi import FastAPI, Request
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
FOLDER_MODEL_CATEGORIES = {"prompt_generator", "LLM"}

# Cache du quota RunPod (récupéré via API au premier appel)
_workspace_quota_gb = None

# ── MOOSEFS CACHE FIX ─────────────────────────────────────

def refresh_dir(path: str):
    """
    Force la relecture du répertoire sur MooseFS/network volumes RunPod.
    MooseFS met en cache les entrées de répertoire côté kernel ; sans ce flush,
    os.scandir/listdir retourne l'ancien état même si des fichiers ont été créés
    par un autre processus (aria2, upload, JupyterLab...).
    Stratégie : open(O_RDONLY|O_DIRECTORY) + fsync → invalide le dentry cache
    pour ce répertoire sans nécessiter de droits root.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        os.fsync(fd)
        os.close(fd)
    except Exception:
        pass

# ── RUNPOD QUOTA ──────────────────────────────────────────

def fetch_runpod_quota():
    """Tailles declarees par RunPod pour ce pod (Go) : volume, container, network volume."""
    global _workspace_quota_gb
    if _workspace_quota_gb is not None:
        return _workspace_quota_gb
    pod_id, api_key = os.environ.get("RUNPOD_POD_ID", ""), os.environ.get("RUNPOD_API_KEY", "")
    if not pod_id or not api_key:
        return {}
    query = f'{{ pod(input: {{podId: "{pod_id}"}}) {{ volumeInGb containerDiskInGb networkVolume {{ size }} }} }}'
    for kw in ({"headers": {"Authorization": f"Bearer {api_key}"}, "url": "https://api.runpod.io/graphql"},
               {"headers": {}, "url": f"https://api.runpod.io/graphql?api_key={api_key}"}):
        try:
            pod = requests.post(kw["url"], json={"query": query}, headers=kw["headers"], timeout=6).json()["data"]["pod"]
            _workspace_quota_gb = {
                "volume": float(pod.get("volumeInGb") or 0),
                "container": float(pod.get("containerDiskInGb") or 0),
                "network": float((pod.get("networkVolume") or {}).get("size") or 0),
            }
            return _workspace_quota_gb
        except Exception:
            continue
    return {}

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
        refresh_dir(base)  # flush MooseFS avant de walker
        for root, dirs, _ in os.walk(base):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for d in dirs:
                rel = os.path.relpath(os.path.join(root, d), base)
                subdirs.append(rel.replace("\\", "/"))
    return sorted(list(set(subdirs)))

@app.get("/fetch-civitai-name")
async def fetch_civitai_name(url: str):
    try:
        is_civitai = "civitai.com" in url or "civitai.red" in url
        if not is_civitai:
            return {"filename": ""}

        hdr = {"User-Agent": "Mozilla/5.0"}
        if CIVITAI_TOKEN:
            hdr["Authorization"] = f"Bearer {CIVITAI_TOKEN}"

        # Cas 1 : URL de telechargement direct /api/download/models/{version_id}
        dl_match = re.search(r"/api/download/models/(\d+)", url)
        if dl_match:
            version_id = dl_match.group(1)
            api_url = f"https://civitai.com/api/v1/model-versions/{version_id}"
            resp = requests.get(api_url, headers=hdr, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                for file in data.get("files", []):
                    if file.get("primary"):
                        return {"filename": file.get("name")}
                if data.get("files"):
                    return {"filename": data["files"][0].get("name")}
            # Fallback : HEAD pour lire Content-Disposition
            try:
                r = requests.head(url, headers=hdr, allow_redirects=True, timeout=8)
                cd = r.headers.get("Content-Disposition", "")
                cd_parts = [p.strip() for p in cd.split(";")]
                fn = ""
                for part in cd_parts:
                    if part.lower().startswith("filename="):
                        fn = part.split("=", 1)[1].strip().strip('"')
                        break
                if fn:
                    return {"filename": fn}
                final_name = unquote(r.url.split("/")[-1].split("?")[0])
                if final_name and "." in final_name:
                    return {"filename": final_name}
            except:
                pass
            return {"filename": ""}

        # Cas 2 : URL de page modele /models/{model_id}
        if "/models/" in url:
            mv_match = re.search(r"modelVersionId=(\d+)", url)
            if mv_match:
                version_id = mv_match.group(1)
                api_url = f"https://civitai.com/api/v1/model-versions/{version_id}"
                resp = requests.get(api_url, headers=hdr, timeout=8)
                if resp.status_code == 200:
                    data = resp.json()
                    for file in data.get("files", []):
                        if file.get("primary"):
                            return {"filename": file.get("name")}
                    if data.get("files"):
                        return {"filename": data["files"][0].get("name")}
            else:
                model_id = url.split("/models/")[1].split("?")[0].split("/")[0]
                api_url = f"https://civitai.com/api/v1/models/{model_id}"
                resp = requests.get(api_url, headers=hdr, timeout=8)
                if resp.status_code == 200:
                    data = resp.json()
                    versions = data.get("modelVersions", [])
                    if versions:
                        for file in versions[0].get("files", []):
                            if file.get("primary"):
                                return {"filename": file.get("name")}
                        if versions[0].get("files"):
                            return {"filename": versions[0]["files"][0].get("name")}

        return {"filename": ""}
    except:
        return {"filename": ""}

# ── WORKFLOW : modeles manquants ─────────────────────────────
WORKFLOWS_DIR = "/workspace/ComfyUI/user/default/workflows"
HF_REPO = "rafatribe/Personnal_Loras"
MODEL_EXT = ('.safetensors', '.gguf', '.pth', '.pt', '.ckpt', '.bin', '.sft', '.onnx')
NODE_FOLDERS = [   # (morceau du type de node, dossier) ; premier qui matche
    ("lora", "loras"), ("vae", "vae"), ("clipvision", "clip_vision"), ("clip_vision", "clip_vision"),
    ("clip", "text_encoders"), ("textencoder", "text_encoders"), ("upscalemodel", "upscale_models"),
    ("latentupscale", "latent_upscale_models"), ("controlnet", "controlnet"), ("modelpatch", "model_patches"),
    ("checkpoint", "checkpoints"), ("gguf", "diffusion_models"), ("unet", "diffusion_models"),
    ("diffusion", "diffusion_models"), ("model", "diffusion_models"),
]
_hf_index = {"t": 0, "files": {}}

def hf_repo_index():
    """nom de fichier -> chemin dans le repo HF prive (cache 10 min)."""
    if time.time() - _hf_index["t"] < 600:
        return _hf_index["files"]
    files = {}
    try:
        from huggingface_hub import HfApi
        for f in HfApi(token=HF_TOKEN or None).list_repo_files(HF_REPO):
            name = f.split("/")[-1]
            if name.lower().endswith(MODEL_EXT):
                # priorite au backup/models/
                if name not in files or f.startswith("backup/models/"):
                    files[name] = f
    except Exception:
        pass
    _hf_index.update(t=time.time(), files=files)
    return files

def catalogue_index():
    idx = {}
    if os.path.exists(CONFIG_PATH):
        cfg = json.load(open(CONFIG_PATH, encoding="utf-8"))
        for env in cfg.values():
            for cat, items in env.items():
                for m in items:
                    idx.setdefault(m["filename"], {"url": m["url"], "path": m.get("path") or cat})
    return idx

def disk_index():
    """nom de fichier -> chemin relatif a models/ (tous dossiers)."""
    idx = {}
    for root, dirs, files in os.walk(BASE_MODELS_PATH, followlinks=True):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for f in files:
            full = os.path.join(root, f)
            try:
                if os.path.getsize(full) > 0:
                    idx.setdefault(f, os.path.relpath(full, BASE_MODELS_PATH))
            except OSError:
                pass
    return idx

def folder_for(node_type):
    t = node_type.lower().replace(" ", "").replace("_", "")
    for key, folder in NODE_FOLDERS:
        if key.replace("_", "") in t:
            return folder
    return ""

def walk_strings(v):
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for x in v.values(): yield from walk_strings(x)
    elif isinstance(v, list):
        for x in v: yield from walk_strings(x)

def models_from_workflow(wf):
    """Liste de {value, folder, url} a partir d'un workflow ComfyUI (nodes + subgraphs)."""
    nodes = list(wf.get("nodes", []))
    for sg in (wf.get("definitions") or {}).get("subgraphs", []):
        nodes += sg.get("nodes", [])
    # liens et dossiers declares dans le workflow (templates ComfyUI), toutes sources confondues
    embedded = {}
    for m in list(wf.get("models") or []) + [m for n in nodes for m in ((n.get("properties") or {}).get("models") or [])]:
        if isinstance(m, dict) and m.get("name"):
            e = embedded.setdefault(m["name"], {})
            e.setdefault("url", m.get("url", ""))
            e.setdefault("directory", m.get("directory", ""))
    found = {}
    for n in nodes:
        if n.get("mode", 0) in (2, 4):      # node muet ou bypass : ignore
            continue
        ntype = n.get("type", "")
        for s in walk_strings(n.get("widgets_values")):
            if not s.lower().endswith(MODEL_EXT) or s.startswith(("http", "/")):
                continue
            value = s.replace("\\", "/")
            emb = embedded.get(value.split("/")[-1]) or {}
            cur = found.setdefault(value, {"value": value, "folder": "", "url": "", "node": ""})
            cur["folder"] = cur["folder"] or emb.get("directory") or folder_for(ntype)
            cur["url"] = cur["url"] or emb.get("url", "")
            if folder_for(ntype):
                cur["node"] = cur["node"] or ntype
    return list(found.values())

def models_from_text(text):
    out = {}
    for m in re.findall(r"[\w\-.+()\[\]/\\]+\.(?:safetensors|gguf|pth|pt|ckpt|bin|sft|onnx)\b", text):
        v = m.strip().replace("\\", "/")
        out.setdefault(v, {"value": v, "folder": "", "url": "", "node": ""})
    return list(out.values())

def hf_size(url):
    try:
        h = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
        r = requests.head(url, headers=h, allow_redirects=False, timeout=6)
        return int(r.headers.get("X-Linked-Size") or r.headers.get("Content-Length") or 0)
    except Exception:
        return 0

def fix_url(url, name):
    """Lien HF dont le fichier ne porte pas le nom attendu (erreur frequente dans les templates) :
    essaie le meme dossier avec le bon nom. Renvoie (url, corrige)."""
    if "huggingface.co" not in url:
        return url, False
    base = unquote(urlparse(url).path.split("/")[-1])
    if base == name:
        return url, False
    cand = url.split("?")[0].rsplit("/", 1)[0] + "/" + name
    return (cand, True) if hf_size(cand) > 0 else (url, False)

@app.get("/workflows")
async def list_workflows():
    res = []
    if os.path.isdir(WORKFLOWS_DIR):
        for root, _, files in os.walk(WORKFLOWS_DIR, followlinks=True):
            for f in files:
                if f.endswith(".json"):
                    res.append(os.path.relpath(os.path.join(root, f), WORKFLOWS_DIR))
    return sorted(res, key=str.lower)

@app.post("/analyze-workflow")
async def analyze_workflow(request: Request):
    data = await request.json()
    text = data.get("text", "")
    if data.get("workflow"):
        p = os.path.realpath(os.path.join(WORKFLOWS_DIR, data["workflow"]))
        text = open(p, encoding="utf-8").read()
    elif data.get("template"):
        raw = template_file(data["template"] + ".json")
        if raw is None:
            return {"error": "Template introuvable (ni dans ComfyUI, ni sur GitHub)."}
        text = raw.decode("utf-8")
    try:
        items = models_from_workflow(json.loads(text))
    except Exception:
        items = models_from_text(text)

    disk, cat, hf = disk_index(), catalogue_index(), hf_repo_index()
    results = []
    for it in items:
        value = it["value"]
        name = value.split("/")[-1]
        sub = "/".join(value.split("/")[:-1])          # sous-dossier tel qu'ecrit dans le workflow
        entry = {"value": value, "filename": name, "node": it["node"], "size": 0, "in_catalogue": name in cat}
        c = cat.get(name)
        folder = it["folder"] or (c["path"].split("/")[0] if c else "")
        if not folder and hf.get(name, "").startswith("backup/models/"):
            folder = hf[name][len("backup/models/"):].split("/")[0]
        entry["path"] = "/".join(x for x in (folder, sub) if x)
        if not sub and c and "/" in c["path"] and c["path"].split("/")[0] == folder:
            entry["path"] = c["path"]                   # sous-dossier du catalogue
        if name in hf:
            entry.update(source="HF perso", url=f"https://huggingface.co/{HF_REPO}/resolve/main/{hf[name]}")
        elif c:
            entry.update(source="catalogue", url=c["url"])
        elif it["url"]:
            entry.update(source="lien du workflow", url=it["url"])
        entry["public_url"] = it["url"] or (c["url"] if c else "")   # lien a ecrire dans le catalogue
        if entry.get("url") and entry.get("source") != "HF perso":
            entry["url"], fixed = fix_url(entry["url"], name)
            if fixed:
                entry["public_url"] = entry["url"]
                entry["url_fixed"] = True
                if c:
                    entry["in_catalogue"] = False      # lien faux dans models.json : a reecrire
        if name in disk:
            entry.update(status="present", where=disk[name])
        elif entry.get("url"):
            entry["status"] = "manquant" if folder else "dossier inconnu"
        else:
            entry["status"] = "introuvable"
        results.append(entry)

    from concurrent.futures import ThreadPoolExecutor
    todo = [e for e in results if e["status"] != "present" and e.get("url", "").startswith("https://huggingface.co")]
    with ThreadPoolExecutor(8) as ex:
        for e, size in zip(todo, ex.map(lambda e: hf_size(e["url"]), todo)):
            e["size"] = size
    order = {"manquant": 0, "dossier inconnu": 1, "introuvable": 2, "present": 3}
    results.sort(key=lambda e: (order.get(e["status"], 9), e["filename"].lower()))
    return results

# ── CATALOGUE DE TEMPLATES COMFYUI ───────────────────────────
COMFY_URL = "http://127.0.0.1:8188"
TEMPLATES_GH = "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates"
_tpl_cache = {"t": 0, "data": None}
_thumb_cache = {}

def template_file(fname):
    """Fichier de template : d'abord le ComfyUI local (version installee), sinon GitHub."""
    for base in (f"{COMFY_URL}/templates", TEMPLATES_GH):
        try:
            r = requests.get(f"{base}/{fname}", timeout=8 if base == COMFY_URL else 20)
            if r.status_code == 200 and r.content:
                return r.content
        except Exception:
            pass
    return None

@app.get("/templates-index")
async def templates_index():
    if _tpl_cache["data"] and time.time() - _tpl_cache["t"] < 3600:
        return _tpl_cache["data"]
    raw = template_file("index.fr.json") or template_file("index.json")
    if raw is None:
        return {"error": "Catalogue indisponible (ComfyUI arrete et GitHub injoignable).", "templates": []}
    out = []
    for cat in json.loads(raw):
        for t in cat.get("templates", []):
            out.append({
                "name": t["name"], "title": t.get("title", t["name"]), "description": t.get("description", ""),
                "category": cat.get("title", ""), "tags": t.get("tags", []), "models": t.get("models", []),
                "size": t.get("size", 0), "api": t.get("openSource") is False or "API" in t.get("tags", []),
                "date": t.get("date", ""), "ext": t.get("mediaSubtype", "webp"),
            })
    data = {"templates": out}
    _tpl_cache.update(t=time.time(), data=data)
    return data

@app.get("/template-thumb/{name}")
async def template_thumb(name: str, ext: str = "webp"):
    from fastapi.responses import Response
    key = f"{name}.{ext}"
    if key not in _thumb_cache:
        raw = template_file(f"{name}-1.{ext}")
        if raw is None:
            return Response(status_code=404)
        try:   # miniature 360 px (les vignettes d'origine font ~1 Mo)
            from PIL import Image
            import io
            im = Image.open(io.BytesIO(raw)); im.thumbnail((360, 360))
            buf = io.BytesIO(); im.convert("RGB").save(buf, "WEBP", quality=80)
            raw, ext = buf.getvalue(), "webp"
        except Exception:
            pass
        _thumb_cache[key] = (raw, ext)
    raw, ext = _thumb_cache[key]
    return Response(raw, media_type=f"image/{ext}", headers={"Cache-Control": "max-age=86400"})

@app.post("/add-to-catalogue")
async def add_to_catalogue(request: Request):
    """Ajoute des modeles au models.json (env choisi) puis synchronise GitHub."""
    data = await request.json()
    env, items = (data.get("env") or "").strip(), data.get("items", [])
    if not env or not items:
        return {"status": "error", "message": "Environnement et modeles obligatoires."}
    cfg = json.load(open(CONFIG_PATH, encoding="utf-8")) if os.path.exists(CONFIG_PATH) else {}
    added = 0
    for it in items:
        path = it["path"].strip("/")
        cat = path.split("/")[0]
        obj = {"name": it["filename"], "url": it["url"], "path": path, "category": cat, "filename": it["filename"]}
        existing = [(l, k) for e in cfg.values() for l in e.values() for k, m in enumerate(l) if m["filename"] == it["filename"]]
        if existing:                     # deja au catalogue (lien a corriger) : mise a jour sur place
            for l, k in existing:
                l[k]["url"] = it["url"]
        else:
            cfg.setdefault(env, {}).setdefault(cat, []).append(obj); added += 1
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4)
    return {"status": "ok", "added": added, "github_sync": "ok" if sync_to_github() else "failed"}

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
    github_synced = sync_to_github()
    return {"status": "ok", "github_sync": "ok" if github_synced else "failed"}

@app.post("/sync-github")
async def sync_github_endpoint():
    result = sync_to_github()
    if result:
        return {"status": "ok"}
    return {"status": "error", "message": "Sync échoué — vérifier GITHUB_TOKEN"}

@app.get("/scan-disk")
async def scan_disk():
    res = {}
    if os.path.exists(BASE_MODELS_PATH):
        refresh_dir(BASE_MODELS_PATH)  # flush racine models avant de scanner
        for entry in os.scandir(BASE_MODELS_PATH):
            if entry.is_dir():
                cat_name = entry.name
                refresh_dir(entry.path)  # flush chaque catégorie
                if cat_name in FOLDER_MODEL_CATEGORIES:
                    folders = []
                    for sub in os.scandir(entry.path):
                        if sub.is_dir():
                            refresh_dir(sub.path)
                            total_size = 0
                            for root, _, filenames in os.walk(sub.path):
                                refresh_dir(root)
                                for f in filenames:
                                    try:
                                        total_size += os.path.getsize(os.path.join(root, f))
                                    except:
                                        pass
                            folders.append({"path": sub.name, "size": total_size, "is_folder": True})
                    res[cat_name] = folders
                else:
                    files = []
                    for root, dirs, filenames in os.walk(entry.path):
                        refresh_dir(root)  # flush chaque sous-dossier parcouru
                        for f in filenames:
                            if any(f.endswith(ext) for ext in ALLOWED_EXTENSIONS):
                                rel_path = os.path.relpath(os.path.join(root, f), entry.path)
                                full_path = os.path.join(root, f)
                                size = os.path.getsize(full_path)
                                files.append({"path": rel_path.replace("\\", "/"), "size": size})
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

    is_civitai = "civitai.com" in url.lower() or "civitai.red" in url.lower()
    is_hf = "huggingface.co" in url.lower()

    headers = [
        "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ]

    final_url = url

    if is_civitai:
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
                sep = "&" if "?" in url else "?"
                final_url = f"{url}{sep}token={CIVITAI_TOKEN}"
        except Exception as e:
            return {"status": "error", "message": f"Impossible de résoudre l'URL Civitai : {e}"}

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
        "min-split-size": "5M",
        "piece-length": "1M",
        "header": headers,
        "check-certificate": "false",
        "max-tries": "5",
        "retry-wait": "3",
        "stream-piece-selector": "geom",
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
            raw_name = d.name or ""
            if raw_name.startswith("http"):
                name = filename_fallback(d)
            else:
                name = raw_name

            if not name or name == "Initialisation...":
                try:
                    parsed = urlparse(d.files[0].uris[0]["uri"] if d.files else "")
                    name = unquote(parsed.path.split("/")[-1]) or "Initialisation..."
                except:
                    name = "Initialisation..."

            status = d.status
            if status == "error":
                status = f"Erreur (Code {d.error_code})"

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
                if eta_secs <= 0 or eta_secs > 2_592_000:
                    eta_str = ""
                elif eta_secs > 3600:
                    eta_str = f"{eta_secs // 3600}h{(eta_secs % 3600) // 60}m"
                elif eta_secs > 60:
                    eta_str = f"{eta_secs // 60}m{eta_secs % 60}s"
                else:
                    eta_str = f"{eta_secs}s"
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

_du_cache = {"t": 0, "bytes": 0}

def workspace_used_bytes():
    """Place occupee dans /workspace (parcours complet, cache 60 s : lent sur stockage reseau)."""
    if time.time() - _du_cache["t"] < 60:
        return _du_cache["bytes"]
    total = 0
    for root, dirs, files in os.walk("/workspace"):
        refresh_dir(root)
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    _du_cache.update(t=time.time(), bytes=total)
    return total

@app.get("/disk-usage")
async def disk_usage():
    """Espace reel de /workspace.
    - /workspace sur le Container Disk (pod sans Volume Disk) : df est fiable.
    - /workspace sur un Volume Disk ou un network volume (MooseFS) : df montre tout le cluster,
      on prend la taille declaree par l'API RunPod et on mesure l'occupe nous-memes."""
    GB = 1_000_000_000
    try:
        q = fetch_runpod_quota() or {}
        same_fs = os.stat("/workspace").st_dev == os.stat("/").st_dev
        if same_fs:
            du = shutil.disk_usage("/workspace")
            total, used, kind = du.total / GB, (du.total - du.free) / GB, "container"
        else:
            total = q.get("network") or q.get("volume") or 0
            used, kind = workspace_used_bytes() / GB, ("network" if q.get("network") else "volume")
        if not total:
            return {"workspace": {"total_gb": None, "used_gb": round(used, 1), "free_gb": None, "used_pct": None, "kind": kind}}
        return {"workspace": {
            "total_gb": round(total, 1), "used_gb": round(used, 1),
            "free_gb": round(max(total - used, 0), 1),
            "used_pct": round(min(used / total * 100, 100), 1), "kind": kind,
        }}
    except Exception as e:
        return {"workspace": None, "error": str(e)}

@app.delete("/delete")
async def delete(cat: str, file: str):
    clean_cat = cat.replace(BASE_MODELS_PATH, "").lstrip("/")
    p = os.path.join(BASE_MODELS_PATH, clean_cat, file)
    if os.path.isdir(p):
        shutil.rmtree(p)
    elif os.path.exists(p):
        os.remove(p)
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
        "--rpc-allow-origin-all=true", "--max-concurrent-downloads=6",
        "--follow-torrent=mem", "--rpc-save-upload-metadata=true",
        "--optimize-concurrent-downloads=false", "-D"
    ])
    uvicorn.run(app, host="0.0.0.0", port=8080)
