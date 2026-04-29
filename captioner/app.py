import os, sys, json, asyncio, shutil, subprocess, time
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════
TRANSCRIBER_PATH = '/workspace/models/acestep-transcriber'
CAPTIONER_PATH   = '/workspace/models/acestep-captioner'
WAV_DIR          = '/workspace/captioner/uploads'
OUTPUT_DIR       = '/workspace/captioner/captions'
TARGET_SR        = 16000
MAX_SECONDS      = 60
HF_TOKEN         = os.environ.get('HF_TOKEN', '')

os.makedirs(WAV_DIR,    exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# STATE GLOBAL
# ═══════════════════════════════════════════════════════════
state = {
    "status": "idle",       # idle | downloading | loading | running | done | error
    "log": [],              # liste de messages
    "progress": 0,          # 0-100
    "current_file": "",
    "total_files": 0,
    "processed": 0,
    "errors": 0,
    "models_ready": False,
    "models_loading": False,
}

transcriber = None
transcriber_proc = None
captioner = None
captioner_proc = None

def log(msg: str, level: str = "info"):
    entry = {"time": time.strftime("%H:%M:%S"), "msg": msg, "level": level}
    state["log"].append(entry)
    print(f"[{entry['time']}] {msg}")

# ═══════════════════════════════════════════════════════════
# TÉLÉCHARGEMENT DES MODÈLES
# ═══════════════════════════════════════════════════════════
def models_present():
    for path in [TRANSCRIBER_PATH, CAPTIONER_PATH]:
        if not os.path.exists(path):
            return False
        safetensors = [f for f in os.listdir(path) if f.endswith('.safetensors')]
        if len(safetensors) < 5:
            return False
    return True

def download_model_aria2(repo_id: str, local_dir: str):
    import requests
    os.makedirs(local_dir, exist_ok=True)
    safetensors = [f for f in os.listdir(local_dir) if f.endswith('.safetensors')]
    if len(safetensors) >= 5:
        log(f"✅ {repo_id} déjà présent")
        return

    log(f"⬇️  Téléchargement {repo_id}...")
    headers = {'Authorization': f'Bearer {HF_TOKEN}'} if HF_TOKEN else {}
    resp  = requests.get(f'https://huggingface.co/api/models/{repo_id}', headers=headers, timeout=15)
    files = [s['rfilename'] for s in resp.json().get('siblings', [])]
    log(f"   {len(files)} fichiers trouvés")

    aria2_headers = [f'Authorization: Bearer {HF_TOKEN}'] if HF_TOKEN else []

    for i, fname in enumerate(files):
        dest = os.path.join(local_dir, fname)
        if os.path.exists(dest):
            continue
        url = f'https://huggingface.co/{repo_id}/resolve/main/{fname}'
        cmd = [
            'aria2c', url, '-d', local_dir, '-o', fname,
            '--max-connection-per-server=16', '--split=16',
            '--min-split-size=5M', '--piece-length=1M',
            '--stream-piece-selector=geom', '--continue=true',
            '--allow-overwrite=true', '--auto-file-renaming=false',
            '--check-certificate=false', '--max-tries=5', '--retry-wait=3', '-q',
        ]
        if aria2_headers:
            cmd += ['--header', aria2_headers[0]]
        log(f"   ⬇️  {fname}...")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            log(f"   ❌ Erreur sur {fname}", "error")

def install_deps():
    log("📦 Installation des dépendances...")
    pkgs = [
        ['torchvision==0.26.0', '--index-url', 'https://download.pytorch.org/whl/cu130'],
        ['transformers==5.5.3'],
        ['accelerate'],
        ['optimum-quanto==0.2.4'],
        ['qwen_omni_utils'],
        ['librosa==0.11.0'],
        ['soundfile'],
        ['sentencepiece'],
        ['scipy==1.12.0'],
    ]
    for pkg in pkgs:
        cmd = [sys.executable, '-m', 'pip', 'install'] + pkg + ['--break-system-packages', '-q']
        r = subprocess.run(cmd, capture_output=True)
        name = pkg[0].split('==')[0]
        if r.returncode == 0:
            log(f"   ✅ {name}")
        else:
            log(f"   ❌ {name}", "error")

async def download_and_load_models():
    global transcriber, transcriber_proc, captioner, captioner_proc
    state["status"] = "downloading"
    state["models_loading"] = True

    # Install deps
    install_deps()

    # Start aria2c
    subprocess.run(['pkill', '-9', 'aria2c'], capture_output=True)
    await asyncio.sleep(1)
    subprocess.Popen([
        'aria2c', '--enable-rpc', '--rpc-listen-all=true',
        '--rpc-allow-origin-all=true', '--max-concurrent-downloads=6', '-D'
    ])
    await asyncio.sleep(2)

    # Download models
    if not models_present():
        log("📦 Téléchargement des modèles ACE-Step...")
        download_model_aria2('ACE-Step/acestep-transcriber', TRANSCRIBER_PATH)
        download_model_aria2('ACE-Step/acestep-captioner',   CAPTIONER_PATH)
    else:
        log("✅ Modèles déjà présents")

    # Load models
    state["status"] = "loading"
    log("🔄 Chargement du transcriber...")

    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    import warnings, logging as lg
    warnings.filterwarnings('ignore')
    lg.disable(lg.WARNING)

    transcriber = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        TRANSCRIBER_PATH, torch_dtype=torch.bfloat16, device_map='cpu'
    )
    transcriber.disable_talker()
    transcriber_proc = Qwen2_5OmniProcessor.from_pretrained(TRANSCRIBER_PATH)
    log("✅ Transcriber chargé")

    log("🔄 Chargement du captioner...")
    captioner = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        CAPTIONER_PATH, torch_dtype=torch.bfloat16, device_map='cpu'
    )
    captioner.disable_talker()
    captioner_proc = Qwen2_5OmniProcessor.from_pretrained(CAPTIONER_PATH)
    log("✅ Captioner chargé")

    state["models_ready"] = True
    state["models_loading"] = False
    state["status"] = "idle"
    log("🚀 Prêt — uploadez vos fichiers WAV", "success")

# ═══════════════════════════════════════════════════════════
# CAPTIONING
# ═══════════════════════════════════════════════════════════
def analyze_audio(audio_path):
    import numpy as np
    import librosa

    MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    KEY_NAMES     = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    MAJOR_PROFILE = np.array(MAJOR_PROFILE)
    MINOR_PROFILE = np.array(MINOR_PROFILE)

    y, sr    = librosa.load(audio_path, sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo, '__len__'): tempo = tempo[0]
    bpm         = int(round(float(tempo)))
    chroma      = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_avg  = chroma.mean(axis=1)
    major_corrs = np.array([np.corrcoef(np.roll(MAJOR_PROFILE, i), chroma_avg)[0,1] for i in range(12)])
    minor_corrs = np.array([np.corrcoef(np.roll(MINOR_PROFILE, i), chroma_avg)[0,1] for i in range(12)])
    bm, bmi     = major_corrs.argmax(), minor_corrs.argmax()
    keyscale    = f'{KEY_NAMES[bm]} major' if major_corrs[bm] >= minor_corrs[bmi] else f'{KEY_NAMES[bmi]} minor'
    onset_env   = librosa.onset.onset_strength(y=y, sr=sr)
    _, beats    = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    if len(beats) >= 8:
        bs  = onset_env[beats]
        acf = np.correlate(bs - bs.mean(), bs - bs.mean(), mode='full')[len(bs)-1:]
        timesig = '3' if (len(acf) > 4 and acf[3] > acf[4] * 1.2) else '4'
    else:
        timesig = '4'
    return {'bpm': bpm, 'keyscale': keyscale, 'timesignature': timesig, 'duration': int(round(duration))}

def run_qwen_audio(model, other_model, processor, audio_data, sr, prompt_text):
    import torch
    other_model.to('cpu')
    torch.cuda.empty_cache()
    model.to('cuda')
    conversation = [{
        'role': 'user',
        'content': [
            {'type': 'audio', 'audio': '<|audio_bos|><|AUDIO|><|audio_eos|>'},
            {'type': 'text',  'text': prompt_text},
        ]
    }]
    text   = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=[audio_data], images=None, videos=None,
                       return_tensors='pt', padding=True, sampling_rate=sr)
    inputs = inputs.to(model.device).to(model.dtype)
    with torch.no_grad():
        text_ids = model.generate(**inputs, return_audio=False, max_new_tokens=512)
    output = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    model.to('cpu')
    torch.cuda.empty_cache()
    result = output[0]
    marker = 'assistant\n'
    if marker in result:
        result = result[result.rfind(marker) + len(marker):]
    return result.strip()

async def run_captioning():
    import soundfile as sf
    import librosa as lb
    import numpy as np
    import torch

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    wav_files = sorted([f for f in os.listdir(WAV_DIR) if f.lower().endswith('.wav')])
    state["total_files"] = len(wav_files)
    state["processed"]   = 0
    state["errors"]      = 0
    state["status"]      = "running"

    log(f"🎵 {len(wav_files)} fichiers WAV trouvés", "success")

    for i, filename in enumerate(wav_files):
        audio_path = os.path.join(WAV_DIR, filename)
        base_name  = os.path.splitext(filename)[0]
        txt_path   = os.path.join(OUTPUT_DIR, base_name + '.txt')
        state["current_file"] = filename
        state["progress"] = int((i / len(wav_files)) * 100)

        log(f"\n🎵 {filename}")

        try:
            analysis = analyze_audio(audio_path)
            log(f"   BPM: {analysis['bpm']} | Key: {analysis['keyscale']} | Time: {analysis['timesignature']}/4 | {analysis['duration']}s")

            audio_data, sr = sf.read(audio_path, dtype='float32')
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            if sr != TARGET_SR:
                audio_data = lb.resample(audio_data, orig_sr=sr, target_sr=TARGET_SR)
            if len(audio_data) > MAX_SECONDS * TARGET_SR:
                audio_data = audio_data[:MAX_SECONDS * TARGET_SR]

            log("   📝 Transcription...")
            lyrics = run_qwen_audio(
                transcriber, captioner, transcriber_proc, audio_data, TARGET_SR,
                '*Task* Transcribe this audio in detail'
            )

            language = 'en'
            if '# Languages' in lyrics and '# Lyrics' in lyrics:
                language = lyrics.split('# Languages')[1].split('# Lyrics')[0].replace('\n','').strip()
                lyrics   = lyrics.split('# Lyrics')[1].strip()

            log("   🎼 Caption...")
            caption = run_qwen_audio(
                captioner, transcriber, captioner_proc, audio_data, TARGET_SR,
                '*Task* Describe this music in detail. Include genre, mood, instrumentation, tempo feel, and vocal style if present.'
            )

            out  = f"<CAPTION>\n{caption}\n</CAPTION>\n"
            out += f"<LYRICS>\n{lyrics}\n</LYRICS>\n"
            out += f"<BPM>{analysis['bpm']}</BPM>\n"
            out += f"<KEYSCALE>{analysis['keyscale']}</KEYSCALE>\n"
            out += f"<TIMESIGNATURE>{analysis['timesignature']}</TIMESIGNATURE>\n"
            out += f"<DURATION>{analysis['duration']}</DURATION>\n"
            out += f"<LANGUAGE>{language}</LANGUAGE>"

            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(out)

            log(f"   ✅ {caption[:120]}...", "success")
            state["processed"] += 1

        except Exception as e:
            import traceback
            log(f"   ❌ Erreur : {e}", "error")
            log(traceback.format_exc(), "error")
            state["errors"] += 1

        await asyncio.sleep(0)

    state["progress"] = 100
    state["status"]   = "done"
    log(f"\n✅ Terminé — {state['processed']} traités, {state['errors']} erreurs", "success")

# ═══════════════════════════════════════════════════════════
# ROUTES API
# ═══════════════════════════════════════════════════════════
@app.on_event("startup")
async def startup():
    asyncio.create_task(download_and_load_models())

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / 'index.html'
    if html_path.exists():
        return html_path.read_text()
    return "<h1>index.html introuvable</h1>"

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
        "log":          state["log"][-100:],
    })

@app.post("/upload")
async def upload_wav(files: list[UploadFile] = File(...)):
    uploaded = []
    for file in files:
        if not file.filename.lower().endswith('.wav'):
            continue
        dest = os.path.join(WAV_DIR, file.filename)
        with open(dest, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        uploaded.append(file.filename)
        log(f"📁 Uploadé : {file.filename}")
    return {"uploaded": uploaded, "count": len(uploaded)}

@app.get("/files")
async def list_files():
    wavs = sorted([f for f in os.listdir(WAV_DIR) if f.lower().endswith('.wav')])
    txts = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.txt')])
    return {"wav_files": wavs, "caption_files": txts}

@app.post("/start")
async def start_captioning(background_tasks: BackgroundTasks):
    if not state["models_ready"]:
        return JSONResponse({"error": "Modèles pas encore chargés"}, status_code=400)
    if state["status"] == "running":
        return JSONResponse({"error": "Captioning déjà en cours"}, status_code=400)
    wav_files = [f for f in os.listdir(WAV_DIR) if f.lower().endswith('.wav')]
    if not wav_files:
        return JSONResponse({"error": "Aucun fichier WAV uploadé"}, status_code=400)
    state["log"] = []
    asyncio.create_task(run_captioning())
    return {"status": "started", "files": len(wav_files)}

@app.get("/stream")
async def stream_log():
    async def event_generator():
        last_idx = 0
        while True:
            logs = state["log"]
            if len(logs) > last_idx:
                for entry in logs[last_idx:]:
                    yield f"data: {json.dumps(entry)}\n\n"
                last_idx = len(logs)
            await asyncio.sleep(0.3)
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/download-captions")
async def download_captions():
    import zipfile, io
    txts = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.txt')]
    if not txts:
        return JSONResponse({"error": "Aucune caption générée"}, status_code=404)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in txts:
            zf.write(os.path.join(OUTPUT_DIR, fname), fname)
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": "attachment; filename=captions.zip"})

@app.delete("/clear")
async def clear_files():
    for f in os.listdir(WAV_DIR):
        os.remove(os.path.join(WAV_DIR, f))
    for f in os.listdir(OUTPUT_DIR):
        os.remove(os.path.join(OUTPUT_DIR, f))
    state["log"] = []
    state["status"] = "idle"
    state["progress"] = 0
    state["processed"] = 0
    state["errors"] = 0
    log("🗑️  Fichiers effacés")
    return {"status": "cleared"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
