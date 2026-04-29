import os, sys, json, asyncio, shutil, subprocess, time, zipfile, io
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

TRANSCRIBER_PATH = '/workspace/models/acestep-transcriber'
CAPTIONER_PATH   = '/workspace/models/acestep-captioner'
DATASETS_DIR     = '/workspace/datasets'
TARGET_SR        = 16000
MAX_SECONDS      = 60
HF_TOKEN         = os.environ.get('HF_TOKEN', '')

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
}

transcriber = None
transcriber_proc = None
captioner = None
captioner_proc = None

def log(msg: str, level: str = "info"):
    entry = {"time": time.strftime("%H:%M:%S"), "msg": msg, "level": level}
    state["log"].append(entry)
    print(f"[{entry['time']}] {msg}")

# ── FILE MANAGER ──────────────────────────────────────────

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
            wav_count = sum(1 for _ in item.rglob('*.wav'))
            if wav_count > 0:
                result.append({
                    "type": "dir",
                    "name": item.name,
                    "path": str(item),
                    "wav_count": wav_count,
                    "children": children,
                })
        elif item.suffix.lower() == '.wav':
            size = item.stat().st_size
            result.append({
                "type": "file",
                "name": item.name,
                "path": str(item),
                "size_mb": round(size / 1024 / 1024, 2),
            })
    return result

def get_captions(base_path: str):
    """Return all .txt caption files recursively under base_path."""
    result = []
    base = Path(base_path)
    if not base.exists():
        return result
    for item in sorted(base.rglob('*.txt')):
        if item.name.startswith('.'):
            continue
        size = item.stat().st_size
        try:
            preview = item.read_text(encoding='utf-8')[:400]
        except Exception:
            preview = ''
        result.append({
            "name": item.name,
            "path": str(item),
            "size_kb": round(size / 1024, 1),
            "rel_path": str(item.relative_to(base)),
            "preview": preview,
        })
    return result

@app.get("/tree")
async def file_tree():
    return JSONResponse(get_tree(DATASETS_DIR))

@app.get("/captions")
async def list_captions():
    return JSONResponse(get_captions(DATASETS_DIR))

@app.delete("/file")
async def delete_file(path: str):
    p = Path(path)
    if not str(p).startswith(DATASETS_DIR):
        return JSONResponse({"error": "Chemin non autorisé"}, status_code=403)
    if p.is_dir():
        shutil.rmtree(p)
    elif p.exists():
        p.unlink()
    return {"status": "deleted"}

@app.post("/delete-many")
async def delete_many(request: Request):
    data = await request.json()
    paths = data.get("paths", [])
    deleted = []
    errors = []
    for path in paths:
        p = Path(path)
        if not str(p).startswith(DATASETS_DIR):
            errors.append(path)
            continue
        try:
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
            deleted.append(path)
        except Exception as e:
            errors.append(path)
    return {"deleted": len(deleted), "errors": errors}

@app.post("/rename")
async def rename_file(request: Request):
    data = await request.json()
    src = Path(data["path"])
    if not str(src).startswith(DATASETS_DIR):
        return JSONResponse({"error": "Chemin non autorisé"}, status_code=403)
    dst = src.parent / data["new_name"]
    src.rename(dst)
    return {"status": "renamed", "new_path": str(dst)}

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

# ── MODELS ────────────────────────────────────────────────

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
        log(f"✅ {repo_id} déjà présent")
        return
    log(f"⬇️  Téléchargement {repo_id}...")
    headers = {'Authorization': f'Bearer {HF_TOKEN}'} if HF_TOKEN else {}
    resp  = requests.get(f'https://huggingface.co/api/models/{repo_id}', headers=headers, timeout=15)
    files = [s['rfilename'] for s in resp.json().get('siblings', [])]
    log(f"   {len(files)} fichiers trouvés")
    aria2_h = [f'Authorization: Bearer {HF_TOKEN}'] if HF_TOKEN else []
    for fname in files:
        dest = os.path.join(local_dir, fname)
        if os.path.exists(dest):
            continue
        url = f'https://huggingface.co/{repo_id}/resolve/main/{fname}'
        cmd = ['aria2c', url, '-d', local_dir, '-o', fname,
               '--max-connection-per-server=16', '--split=16',
               '--min-split-size=5M', '--piece-length=1M',
               '--stream-piece-selector=geom', '--continue=true',
               '--allow-overwrite=true', '--auto-file-renaming=false',
               '--check-certificate=false', '--max-tries=5', '--retry-wait=3', '-q']
        if aria2_h:
            cmd += ['--header', aria2_h[0]]
        log(f"   ⬇️  {fname}...")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            log(f"   ❌ Erreur sur {fname}", "error")

def install_deps():
    log("📦 Installation des dépendances...")
    pkgs = [
        ['torchvision==0.26.0', '--index-url', 'https://download.pytorch.org/whl/cu130'],
        ['transformers==5.5.3'], ['accelerate'], ['optimum-quanto==0.2.4'],
        ['qwen_omni_utils'], ['librosa==0.11.0'], ['soundfile'],
        ['sentencepiece'], ['scipy==1.12.0'],
    ]
    for pkg in pkgs:
        cmd = [sys.executable, '-m', 'pip', 'install'] + pkg + ['--break-system-packages', '-q']
        r = subprocess.run(cmd, capture_output=True)
        log(f"   {'✅' if r.returncode == 0 else '❌'} {pkg[0].split('==')[0]}")

async def download_and_load_models():
    global transcriber, transcriber_proc, captioner, captioner_proc
    state["status"] = "downloading"
    state["models_loading"] = True
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
    log("🔄 Chargement du transcriber...")
    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    import warnings, logging as lg
    warnings.filterwarnings('ignore')
    lg.disable(lg.WARNING)
    transcriber = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        TRANSCRIBER_PATH, torch_dtype=torch.bfloat16, device_map='cpu')
    transcriber.disable_talker()
    transcriber_proc = Qwen2_5OmniProcessor.from_pretrained(TRANSCRIBER_PATH)
    log("✅ Transcriber chargé")
    log("🔄 Chargement du captioner...")
    captioner = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        CAPTIONER_PATH, torch_dtype=torch.bfloat16, device_map='cpu')
    captioner.disable_talker()
    captioner_proc = Qwen2_5OmniProcessor.from_pretrained(CAPTIONER_PATH)
    log("✅ Captioner chargé")
    state["models_ready"] = True
    state["models_loading"] = False
    state["status"] = "idle"
    log("🚀 Prêt — sélectionnez des fichiers WAV et lancez le captioning", "success")

# ── CAPTIONING ────────────────────────────────────────────

def analyze_audio(audio_path):
    import numpy as np, librosa
    MP = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    mp = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
    KN = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    y, sr    = librosa.load(audio_path, sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo, '__len__'): tempo = tempo[0]
    bpm   = int(round(float(tempo)))
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    mc = np.array([np.corrcoef(np.roll(MP,i), chroma)[0,1] for i in range(12)])
    nc = np.array([np.corrcoef(np.roll(mp,i), chroma)[0,1] for i in range(12)])
    bm, bn = mc.argmax(), nc.argmax()
    keyscale = f'{KN[bm]} major' if mc[bm] >= nc[bn] else f'{KN[bn]} minor'
    oe = librosa.onset.onset_strength(y=y, sr=sr)
    _, beats = librosa.beat.beat_track(onset_envelope=oe, sr=sr)
    if len(beats) >= 8:
        bs = oe[beats]
        acf = np.correlate(bs-bs.mean(), bs-bs.mean(), mode='full')[len(bs)-1:]
        timesig = '3' if (len(acf) > 4 and acf[3] > acf[4]*1.2) else '4'
    else:
        timesig = '4'
    return {'bpm': bpm, 'keyscale': keyscale, 'timesignature': timesig, 'duration': int(round(duration))}

def run_qwen_audio(model, other, processor, audio_data, sr, prompt):
    import torch
    other.to('cpu')
    torch.cuda.empty_cache()
    model.to('cuda')
    conv = [{'role':'user','content':[
        {'type':'audio','audio':'<|audio_bos|><|AUDIO|><|audio_eos|>'},
        {'type':'text','text': prompt},
    ]}]
    text   = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=[audio_data], images=None, videos=None,
                       return_tensors='pt', padding=True, sampling_rate=sr)
    inputs = inputs.to(model.device).to(model.dtype)
    with torch.no_grad():
        ids = model.generate(**inputs, return_audio=False, max_new_tokens=512)
    out = processor.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    model.to('cpu')
    torch.cuda.empty_cache()
    marker = 'assistant\n'
    if marker in out:
        out = out[out.rfind(marker)+len(marker):]
    return out.strip()

async def run_captioning():
    import soundfile as sf, librosa as lb
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    wav_paths  = state["selected_files"]
    output_dir = state["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    state["total_files"] = len(wav_paths)
    state["processed"]   = 0
    state["errors"]      = 0
    state["status"]      = "running"
    log(f"🎵 {len(wav_paths)} fichiers à traiter", "success")
    for i, audio_path in enumerate(wav_paths):
        filename  = os.path.basename(audio_path)
        base_name = os.path.splitext(filename)[0]
        txt_path  = os.path.join(output_dir, base_name + '.txt')
        state["current_file"] = filename
        state["progress"] = int((i / len(wav_paths)) * 100)
        log(f"\n🎵 {filename}")
        try:
            analysis = analyze_audio(audio_path)
            log(f"   BPM: {analysis['bpm']} | Key: {analysis['keyscale']} | {analysis['timesignature']}/4 | {analysis['duration']}s")
            audio_data, sr = sf.read(audio_path, dtype='float32')
            if audio_data.ndim > 1: audio_data = audio_data.mean(axis=1)
            if sr != TARGET_SR: audio_data = lb.resample(audio_data, orig_sr=sr, target_sr=TARGET_SR)
            if len(audio_data) > MAX_SECONDS * TARGET_SR:
                audio_data = audio_data[:MAX_SECONDS * TARGET_SR]
            log("   📝 Transcription...")
            lyrics = run_qwen_audio(transcriber, captioner, transcriber_proc, audio_data, TARGET_SR,
                                    '*Task* Transcribe this audio in detail')
            language = 'en'
            if '# Languages' in lyrics and '# Lyrics' in lyrics:
                language = lyrics.split('# Languages')[1].split('# Lyrics')[0].replace('\n','').strip()
                lyrics   = lyrics.split('# Lyrics')[1].strip()
            log("   🎼 Caption...")
            caption = run_qwen_audio(captioner, transcriber, captioner_proc, audio_data, TARGET_SR,
                                     '*Task* Describe this music in detail. Include genre, mood, instrumentation, tempo feel, and vocal style if present.')
            out  = f"<CAPTION>\n{caption}\n</CAPTION>\n"
            out += f"<LYRICS>\n{lyrics}\n</LYRICS>\n"
            out += f"<BPM>{analysis['bpm']}</BPM>\n"
            out += f"<KEYSCALE>{analysis['keyscale']}</KEYSCALE>\n"
            out += f"<TIMESIGNATURE>{analysis['timesignature']}</TIMESIGNATURE>\n"
            out += f"<DURATION>{analysis['duration']}</DURATION>\n"
            out += f"<LANGUAGE>{language}</LANGUAGE>"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(out)
            log(f"   ✅ {caption[:100]}...", "success")
            state["processed"] += 1
        except Exception as e:
            import traceback
            log(f"   ❌ {e}", "error")
            log(traceback.format_exc(), "error")
            state["errors"] += 1
        await asyncio.sleep(0)
    state["progress"] = 100
    state["status"]   = "done"
    log(f"\n✅ Terminé — {state['processed']} traités, {state['errors']} erreurs", "success")

# ── ROUTES ────────────────────────────────────────────────

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
        "status": state["status"], "models_ready": state["models_ready"],
        "progress": state["progress"], "current_file": state["current_file"],
        "total_files": state["total_files"], "processed": state["processed"],
        "errors": state["errors"], "log": state["log"][-150:],
        "output_dir": state["output_dir"],
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
    if not wav_paths:
        return JSONResponse({"error": "Aucun fichier sélectionné"}, status_code=400)
    if not output_dir:
        return JSONResponse({"error": "Dossier de sortie non spécifié"}, status_code=400)
    state["selected_files"] = wav_paths
    state["output_dir"]     = output_dir
    state["log"]            = []
    asyncio.create_task(run_captioning())
    return {"status": "started", "files": len(wav_paths)}

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
    uvicorn.run(app, host="0.0.0.0", port=7860)
