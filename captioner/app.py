"""
ACE-Step Captioner — app.py
Supports WAV, MP3, FLAC, OGG, AIFF, M4A.
Heavy inference runs in a separate Process via ProcessPoolExecutor.
Torch cu124 — compatible with NVIDIA driver >= 550 (RunPod RTX 4090, A100, H100, L40S).
"""
import os, sys, json, asyncio, shutil, subprocess, time, zipfile, io
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

sys.stdout.reconfigure(line_buffering=True)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

TRANSCRIBER_PATH = '/workspace/models/acestep-transcriber'
CAPTIONER_PATH   = '/workspace/models/acestep-captioner'
DATASETS_DIR     = '/workspace/datasets'
TARGET_SR        = 16000
HF_TOKEN         = os.environ.get('HF_TOKEN', '')

AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aif', '.m4a'}

# ── PRESETS ───────────────────────────────────────────────────────────────────
PRESETS = {
    "generic": {
        "label":        "Générique",
        "threshold_db": -40,
        "min_skip":     0,
        "max_seconds":  120,
        "skip_ratio":   0.0,
    },
    "techno": {
        "label":        "Techno / Tek House",
        "threshold_db": -18,
        "min_skip":     20,
        "max_seconds":  120,
        "skip_ratio":   0.25,
    },
    "ambient": {
        "label":        "Ambient / Cinématique",
        "threshold_db": -30,
        "min_skip":     5,
        "max_seconds":  180,
        "skip_ratio":   0.1,
    },
}

os.makedirs(DATASETS_DIR, exist_ok=True)

state = {
    "status": "idle",
    "log": [],
    "progress": 0,
    "current_file": "",
    "total_files": 0,
    "processed": 0,
    "errors": 0,
    "models_ready": False,
    "models_loading": False,
    "selected_files": [],
    "output_dir": "",
    "preset": "generic",
}

process_pool = ProcessPoolExecutor(max_workers=1)

# ── LOGGING ───────────────────────────────────────────────────────────────────

def log(msg: str, level: str = "info"):
    entry = {"time": time.strftime("%H:%M:%S"), "msg": msg, "level": level}
    state["log"].append(entry)
    print(f"[{entry['time']}] {msg}", flush=True)

# ── FILE MANAGER ──────────────────────────────────────────────────────────────

def is_audio(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTS

def get_tree(base_path: str):
    result = []
    base = Path(base_path)
    if not base.exists():
        return result
    for item in sorted(base.iterdir()):
        if item.name.startswith('.'):
            continue
        if item.is_dir():
            children = get_tree(str(item))
            audio_count = sum(1 for f in item.rglob('*') if f.is_file() and is_audio(f))
            if audio_count > 0:
                result.append({
                    "type": "dir", "name": item.name, "path": str(item),
                    "wav_count": audio_count, "children": children,
                })
        elif is_audio(item):
            result.append({
                "type": "file", "name": item.name, "path": str(item),
                "size_mb": round(item.stat().st_size / 1024 / 1024, 2),
                "ext": item.suffix.lower().lstrip('.'),
            })
    return result

def get_captions(base_path: str):
    result = []
    base = Path(base_path)
    if not base.exists():
        return result
    for item in sorted(base.rglob('*.txt')):
        if item.name.startswith('.'):
            continue
        try:
            preview = item.read_text(encoding='utf-8')[:400]
        except Exception:
            preview = ''
        result.append({
            "name": item.name, "path": str(item),
            "size_kb": round(item.stat().st_size / 1024, 1),
            "rel_path": str(item.relative_to(base)), "preview": preview,
        })
    return result

@app.get("/tree")
async def file_tree():
    return JSONResponse(get_tree(DATASETS_DIR))

@app.get("/captions")
async def list_captions():
    return JSONResponse(get_captions(DATASETS_DIR))

@app.get("/presets")
async def list_presets():
    return JSONResponse({k: {"label": v["label"]} for k, v in PRESETS.items()})

@app.delete("/file")
async def delete_file(path: str):
    p = Path(path)
    if not str(p).startswith(DATASETS_DIR):
        return JSONResponse({"error": "Chemin non autorisé"}, status_code=403)
    if p.is_dir():   shutil.rmtree(p)
    elif p.exists(): p.unlink()
    return {"status": "deleted"}

@app.post("/delete-many")
async def delete_many(request: Request):
    data = await request.json()
    deleted, errors = [], []
    for path in data.get("paths", []):
        p = Path(path)
        if not str(p).startswith(DATASETS_DIR):
            errors.append(path); continue
        try:
            if p.is_dir():   shutil.rmtree(p)
            elif p.exists(): p.unlink()
            deleted.append(path)
        except Exception:
            errors.append(path)
    return {"deleted": len(deleted), "errors": errors}

@app.post("/rename")
async def rename_file(request: Request):
    data = await request.json()
    src = Path(data["path"])
    if not str(src).startswith(DATASETS_DIR):
        return JSONResponse({"error": "Chemin non autorisé"}, status_code=403)
    src.rename(src.parent / data["new_name"])
    return {"status": "renamed"}

@app.post("/mkdir")
async def make_dir(request: Request):
    data = await request.json()
    path = Path(data["path"])
    if not str(path).startswith(DATASETS_DIR):
        return JSONResponse({"error": "Chemin non autorisé"}, status_code=403)
    path.mkdir(parents=True, exist_ok=True)
    return {"status": "created"}

@app.post("/upload-to")
async def upload_to(target_dir: str, files: list[UploadFile] = File(...)):
    p = Path(target_dir)
    if not str(p).startswith(DATASETS_DIR):
        return JSONResponse({"error": "Chemin non autorisé"}, status_code=403)
    p.mkdir(parents=True, exist_ok=True)
    uploaded = []
    for file in files:
        dest = p / file.filename
        with open(dest, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        uploaded.append(file.filename)
        log(f"📁 Uploadé : {file.filename}")
    return {"uploaded": uploaded, "count": len(uploaded)}

# ── MODEL SETUP ───────────────────────────────────────────────────────────────

def check_driver():
    """Returns (ok, driver_version, cuda_version) — fails fast if driver too old."""
    r = subprocess.run(['nvidia-smi', '--query-gpu=driver_version',
                        '--format=csv,noheader'], capture_output=True, text=True)
    version = r.stdout.strip().split('\n')[0].strip()
    try:
        major = int(version.split('.')[0])
        ok = major >= 550  # minimum for CUDA 12.4 / cu124
    except Exception:
        ok, major = False, 0
    return ok, version

def models_present():
    for path in [TRANSCRIBER_PATH, CAPTIONER_PATH]:
        if not os.path.exists(path):
            return False
        if len([f for f in os.listdir(path) if f.endswith('.safetensors')]) < 5:
            return False
    return True

def download_model_aria2(repo_id: str, local_dir: str):
    import requests
    os.makedirs(local_dir, exist_ok=True)
    if len([f for f in os.listdir(local_dir) if f.endswith('.safetensors')]) >= 5:
        log(f"✅ {repo_id} déjà présent"); return
    log(f"⬇️  Téléchargement {repo_id}...")
    headers = {'Authorization': f'Bearer {HF_TOKEN}'} if HF_TOKEN else {}
    resp  = requests.get(f'https://huggingface.co/api/models/{repo_id}', headers=headers, timeout=15)
    files = [s['rfilename'] for s in resp.json().get('siblings', [])]
    log(f"   {len(files)} fichiers trouvés")
    for fname in files:
        dest = os.path.join(local_dir, fname)
        if os.path.exists(dest): continue
        url = f'https://huggingface.co/{repo_id}/resolve/main/{fname}'
        cmd = ['aria2c', url, '-d', local_dir, '-o', fname,
               '--max-connection-per-server=16', '--split=16', '--min-split-size=5M',
               '--piece-length=1M', '--stream-piece-selector=geom', '--continue=true',
               '--allow-overwrite=true', '--auto-file-renaming=false',
               '--check-certificate=false', '--max-tries=5', '--retry-wait=3', '-q']
        if HF_TOKEN:
            cmd += ['--header', f'Authorization: Bearer {HF_TOKEN}']
        log(f"   ⬇️  {fname}...")
        if subprocess.run(cmd).returncode != 0:
            log(f"   ❌ Erreur sur {fname}", "error")

def install_deps():
    log("📦 Installation des dépendances...")
    pkgs = [
        # cu124 = CUDA 12.4 — compatible driver >= 550 (RTX 4090, A100, H100, L40S on RunPod)
        ['torch==2.5.1', 'torchvision==0.20.1',
         '--index-url', 'https://download.pytorch.org/whl/cu124'],
        ['transformers==5.5.3'], ['accelerate'], ['optimum-quanto==0.2.4'],
        ['qwen_omni_utils'], ['librosa==0.11.0'], ['soundfile'],
        ['sentencepiece'], ['scipy==1.12.0'], ['pydub'],
    ]
    for pkg in pkgs:
        r = subprocess.run(
            [sys.executable, '-m', 'pip', 'install'] + pkg + ['--break-system-packages', '-q'],
            capture_output=True)
        log(f"   {'✅' if r.returncode==0 else '❌'} {pkg[0].split('==')[0]}")

async def download_and_load_models():
    state["status"] = "downloading"
    state["models_loading"] = True

    # Driver check first — fail fast before wasting time downloading
    ok, driver_ver = check_driver()
    if not ok:
        msg = f"❌ Driver NVIDIA trop ancien ({driver_ver}) — minimum requis : 550.x (CUDA 12.4). Changez de pod."
        log(msg, "error")
        state["status"] = "error"
        state["models_loading"] = False
        return

    log(f"✅ Driver OK : {driver_ver}")
    install_deps()

    subprocess.run(['pkill', '-9', 'aria2c'], capture_output=True)
    await asyncio.sleep(1)
    subprocess.Popen(['aria2c', '--enable-rpc', '--rpc-listen-all=true',
                      '--rpc-allow-origin-all=true', '--max-concurrent-downloads=6', '-D'])
    await asyncio.sleep(2)

    if not models_present():
        log("📦 Téléchargement des modèles ACE-Step...")
        download_model_aria2('ACE-Step/acestep-transcriber', TRANSCRIBER_PATH)
        download_model_aria2('ACE-Step/acestep-captioner',   CAPTIONER_PATH)
    else:
        log("✅ Modèles déjà présents")

    state["status"] = "loading"
    log("🔄 Chargement des modèles en mémoire…")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(process_pool, _worker_load_models)
    state["models_ready"]   = True
    state["models_loading"] = False
    state["status"]         = "idle"
    log("🚀 Prêt — sélectionnez des fichiers audio et lancez le captioning", "success")

# ── WORKER PROCESS ────────────────────────────────────────────────────────────

_w_transcriber      = None
_w_transcriber_proc = None
_w_captioner        = None
_w_captioner_proc   = None

def _worker_load_models():
    global _w_transcriber, _w_transcriber_proc, _w_captioner, _w_captioner_proc
    if _w_transcriber is not None:
        return
    import torch, warnings, logging as lg
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    warnings.filterwarnings('ignore')
    lg.disable(lg.WARNING)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    _w_transcriber = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        TRANSCRIBER_PATH, torch_dtype=torch.bfloat16, device_map='cpu')
    _w_transcriber.disable_talker()
    _w_transcriber_proc = Qwen2_5OmniProcessor.from_pretrained(TRANSCRIBER_PATH)
    _w_captioner = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        CAPTIONER_PATH, torch_dtype=torch.bfloat16, device_map='cpu')
    _w_captioner.disable_talker()
    _w_captioner_proc = Qwen2_5OmniProcessor.from_pretrained(CAPTIONER_PATH)


def _load_audio(audio_path: str) -> tuple:
    import numpy as np, librosa
    audio_data, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True, dtype=np.float32)
    return audio_data, TARGET_SR


def _find_drop(audio_data, sr: int, threshold_db: float, min_skip: int, skip_ratio: float) -> int:
    import numpy as np, librosa
    min_skip_samples = min_skip * sr
    rms = librosa.feature.rms(y=audio_data, frame_length=2048, hop_length=512)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-9, ref=np.max)
    frames_above = np.where(rms_db > threshold_db)[0]
    if len(frames_above) > 0:
        drop_sample = librosa.frames_to_samples(int(frames_above[0]), hop_length=512)
        start = max(drop_sample, min_skip_samples)
    else:
        start = max(int(len(audio_data) * skip_ratio), min_skip_samples)
    return int(start)


def _worker_analyze(audio_path: str) -> dict:
    import numpy as np, librosa
    MP = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    mp = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
    KN = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    y, sr    = librosa.load(audio_path, sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = int(round(float(tempo[0] if hasattr(tempo,'__len__') else tempo)))
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    mc = np.array([np.corrcoef(np.roll(MP,i), chroma)[0,1] for i in range(12)])
    nc = np.array([np.corrcoef(np.roll(mp,i), chroma)[0,1] for i in range(12)])
    bm, bn = mc.argmax(), nc.argmax()
    keyscale = f'{KN[bm]} major' if mc[bm] >= nc[bn] else f'{KN[bn]} minor'
    oe = librosa.onset.onset_strength(y=y, sr=sr)
    _, beats = librosa.beat.beat_track(onset_envelope=oe, sr=sr)
    timesig = '4'
    if len(beats) >= 8:
        bs = oe[beats]
        acf = np.correlate(bs-bs.mean(), bs-bs.mean(), mode='full')[len(bs)-1:]
        if len(acf) > 4 and acf[3] > acf[4]*1.2:
            timesig = '3'
    return {'bpm': bpm, 'keyscale': keyscale, 'timesignature': timesig, 'duration': int(round(duration))}


def _worker_qwen(use_transcriber: bool, audio_data, sr: int, prompt: str) -> str:
    import torch
    global _w_transcriber, _w_transcriber_proc, _w_captioner, _w_captioner_proc
    if use_transcriber:
        model, other, proc = _w_transcriber, _w_captioner, _w_transcriber_proc
    else:
        model, other, proc = _w_captioner, _w_transcriber, _w_captioner_proc
    other.to('cpu')
    torch.cuda.empty_cache()
    model.to('cuda')
    conv = [{'role':'user','content':[
        {'type':'audio','audio':'<|audio_bos|><|AUDIO|><|audio_eos|>'},
        {'type':'text','text': prompt},
    ]}]
    text   = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=text, audio=[audio_data], images=None, videos=None,
                  return_tensors='pt', padding=True, sampling_rate=sr)
    inputs = inputs.to(model.device).to(model.dtype)
    with torch.no_grad():
        ids = model.generate(**inputs, return_audio=False, max_new_tokens=512)
    out = proc.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    model.to('cpu')
    torch.cuda.empty_cache()
    marker = 'assistant\n'
    if marker in out:
        out = out[out.rfind(marker)+len(marker):]
    return out.strip()


def _worker_process_file(audio_path: str, output_dir: str, preset_key: str) -> dict:
    import traceback as tb
    try:
        _worker_load_models()

        preset       = PRESETS.get(preset_key, PRESETS["generic"])
        threshold_db = preset["threshold_db"]
        min_skip     = preset["min_skip"]
        max_seconds  = preset["max_seconds"]
        skip_ratio   = preset["skip_ratio"]

        analysis = _worker_analyze(audio_path)

        audio_data, sr = _load_audio(audio_path)

        start_sample = _find_drop(audio_data, TARGET_SR, threshold_db, min_skip, skip_ratio)
        end_sample   = start_sample + max_seconds * TARGET_SR
        audio_data   = audio_data[start_sample:end_sample]
        start_sec    = round(start_sample / TARGET_SR, 1)

        lyrics = _worker_qwen(True, audio_data, TARGET_SR,
                              '*Task* Transcribe this audio in detail')
        language = 'en'
        if '# Languages' in lyrics and '# Lyrics' in lyrics:
            language = lyrics.split('# Languages')[1].split('# Lyrics')[0].replace('\n','').strip()
            lyrics   = lyrics.split('# Lyrics')[1].strip()

        caption = _worker_qwen(False, audio_data, TARGET_SR,
                               '*Task* Describe this music in detail. Include genre, mood, '
                               'instrumentation, tempo feel, and vocal style if present.')

        out  = f"<CAPTION>\n{caption}\n</CAPTION>\n"
        out += f"<LYRICS>\n{lyrics}\n</LYRICS>\n"
        out += f"<BPM>{analysis['bpm']}</BPM>\n"
        out += f"<KEYSCALE>{analysis['keyscale']}</KEYSCALE>\n"
        out += f"<TIMESIGNATURE>{analysis['timesignature']}</TIMESIGNATURE>\n"
        out += f"<DURATION>{analysis['duration']}</DURATION>\n"
        out += f"<LANGUAGE>{language}</LANGUAGE>\n"
        out += f"<PRESET>{preset_key}</PRESET>\n"
        out += f"<ANALYSIS_START>{start_sec}s</ANALYSIS_START>"

        base = os.path.splitext(os.path.basename(audio_path))[0]
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, base + '.txt'), 'w', encoding='utf-8') as f:
            f.write(out)

        return {
            "ok": True,
            "bpm": analysis['bpm'], "keyscale": analysis['keyscale'],
            "timesig": analysis['timesignature'], "duration": analysis['duration'],
            "start_sec": start_sec,
            "caption_preview": caption[:120],
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "traceback": tb.format_exc()}

# ── CAPTIONING ORCHESTRATOR ───────────────────────────────────────────────────

async def run_captioning():
    wav_paths  = state["selected_files"]
    output_dir = state["output_dir"]
    preset_key = state["preset"]
    preset     = PRESETS.get(preset_key, PRESETS["generic"])

    state["total_files"] = len(wav_paths)
    state["processed"]   = 0
    state["errors"]      = 0
    state["status"]      = "running"

    log(f"🚀 Batch — {len(wav_paths)} fichier(s) | preset: {preset['label']}", "success")
    log(f"   threshold: {preset['threshold_db']}dB | min_skip: {preset['min_skip']}s | max: {preset['max_seconds']}s")

    loop = asyncio.get_event_loop()

    for i, audio_path in enumerate(wav_paths):
        filename = os.path.basename(audio_path)
        state["current_file"] = filename
        state["progress"] = int((i / len(wav_paths)) * 100)
        log(f"🎵 [{i+1}/{len(wav_paths)}] {filename}")

        result = await loop.run_in_executor(
            process_pool, _worker_process_file, audio_path, output_dir, preset_key
        )

        if result["ok"]:
            log(f"   ▶ Depuis {result['start_sec']}s | BPM:{result['bpm']} | {result['keyscale']} | {result['duration']}s total", "success")
            log(f"   📝 {result['caption_preview']}…", "success")
            state["processed"] += 1
        else:
            log(f"   ❌ {result['error']}", "error")
            log(result.get("traceback",""), "error")
            state["errors"] += 1

    state["progress"]     = 100
    state["status"]       = "done"
    state["current_file"] = ""
    log(f"✅ Terminé — {state['processed']} traités, {state['errors']} erreurs", "success")

# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    asyncio.create_task(download_and_load_models())

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / 'index.html'
    return html_path.read_text() if html_path.exists() else "<h1>index.html introuvable</h1>"

@app.get("/status")
async def get_status():
    return JSONResponse({
        "status":       state["status"],
        "models_ready": state["models_ready"],
        "progress":     state["progress"],
        "current_file": state["current_file"],
        "total_files":  state["total_files"],
        "processed":    state["processed"],
        "errors":       state["errors"],
        "log":          state["log"][-200:],
        "output_dir":   state["output_dir"],
        "preset":       state["preset"],
    })

@app.post("/start")
async def start_captioning(request: Request):
    if not state["models_ready"]:
        return JSONResponse({"error": "Modèles pas encore chargés"}, status_code=400)
    if state["status"] == "running":
        return JSONResponse({"error": "Captioning déjà en cours"}, status_code=400)
    data = await request.json()
    wav_paths  = data.get("files", [])
    output_dir = data.get("output_dir", "")
    preset_key = data.get("preset", "generic")
    if not wav_paths:
        return JSONResponse({"error": "Aucun fichier sélectionné"}, status_code=400)
    if not output_dir:
        return JSONResponse({"error": "Dossier de sortie non spécifié"}, status_code=400)
    if preset_key not in PRESETS:
        preset_key = "generic"

    state["selected_files"] = wav_paths
    state["output_dir"]     = output_dir
    state["preset"]         = preset_key
    state["log"]            = []
    state["progress"]       = 0
    state["processed"]      = 0
    state["errors"]         = 0

    log(f"🚀 Batch reçu — {len(wav_paths)} fichier(s) | preset: {PRESETS[preset_key]['label']}", "success")
    asyncio.create_task(run_captioning())
    return {"status": "started", "files": len(wav_paths), "preset": preset_key}

@app.get("/download-captions")
async def download_captions():
    output_dir = state["output_dir"]
    if not output_dir or not os.path.exists(output_dir):
        return JSONResponse({"error": "Aucun dossier de sortie"}, status_code=404)
    txts = [f for f in os.listdir(output_dir) if f.endswith('.txt')]
    if not txts:
        return JSONResponse({"error": "Aucune caption générée"}, status_code=404)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in txts:
            zf.write(os.path.join(output_dir, fname), fname)
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": "attachment; filename=captions.zip"})

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    uvicorn.run(app, host="0.0.0.0", port=7860)
