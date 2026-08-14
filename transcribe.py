#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "faster-whisper>=1.1",
#   "google-api-python-client>=2.100",
#   "google-auth-oauthlib>=1.2",
#   "nvidia-cublas-cu12; sys_platform == 'win32'",
#   "nvidia-cudnn-cu12>=9,<10; sys_platform == 'win32'",
# ]
# ///
"""Transcrit des fichiers audio/vidéo Google Drive en Google Docs via Whisper.

Usage:
    uv run transcribe.py <dossier-drive (URL ou ID)> [options]
    uv run transcribe.py <fichier-ou-dossier-local> [options]

Pré-requis : ~/.transcribe/credentials.json (client OAuth "Desktop app",
voir README.md). Le premier lancement ouvre le navigateur pour autoriser
l'accès à Google Drive ; le jeton est ensuite mis en cache.
"""

import argparse
import io
import os
import re
import sys
import tempfile
import time
from pathlib import Path

# Xet télécharge via un cache intermédiaire, ce qui rend la progression
# invisible (et ses barres tqdm ne s'affichent pas de façon fiable) ;
# le mode HTTP classique écrit directement dans le cache du modèle.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Windows : console et sous-processus utilisent cp1252 par défaut, qui ne
# peut pas encoder certains caractères du script (→, ✓, œ…) — tout en UTF-8.
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

CONFIG_DIR = Path.home() / ".transcribe"
CREDENTIALS_FILE = CONFIG_DIR / "credentials.json"
TOKEN_FILE = CONFIG_DIR / "token.json"
SCOPES = ["https://www.googleapis.com/auth/drive"]

GOOGLE_DOC_MIME = "application/vnd.google-apps.document"
# Métadonnée Drive (appProperties) marquant un Doc déjà corrigé par Claude
CORRECTED_PROP = "transcribeCorrected"

# Extensions reconnues en mode local (source = fichier/dossier sur disque)
MEDIA_EXTENSIONS = {
    ".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma", ".aiff",
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg",
}

# Pause entre deux segments au-delà de laquelle on démarre un paragraphe
PARAGRAPH_GAP_SECONDS = 2.0
# Longueur au-delà de laquelle on coupe le paragraphe à la prochaine pause
PARAGRAPH_MAX_CHARS = 1200

# ---------------------------------------------------------------------------
# Correction Claude (post-traitement, via Claude Code en mode headless)
# ---------------------------------------------------------------------------

# Le glossaire du domaine vit dans glossaire.txt à côté du script
# (versionné dans le repo), complété par ~/.transcribe/glossaire.txt
# pour les ajouts personnels locaux. Un terme par ligne, # = commentaire.
SCRIPT_GLOSSARY_FILE = Path(__file__).resolve().parent / "glossaire.txt"
GLOSSARY_FILE = CONFIG_DIR / "glossaire.txt"

CORRECTION_PROMPT = """\
Tu corriges la transcription automatique (Whisper) d'un cours oral en \
français donné dans le cadre d'une formation de somatopathie — ostéopathie \
douce selon la méthode Poyet.

Contexte technique : le cours porte sur la somatopathie et l'ostéopathie \
informationnelle méthode Poyet (Maurice-Raymond Poyet) — le mouvement \
respiratoire primaire (MRP), l'axe crânio-sacré (crâne, sacrum, dure-mère, \
membranes de tension réciproque), l'écoute tissulaire et les corrections \
douces par induction, l'anatomie (os du crâne, rachis, bassin, membres, \
fascias, viscères, système nerveux), la biomécanique, la palpation, ainsi \
que l'énergétique chinoise (méridiens, vaisseau gouverneur et vaisseau \
conception) sur laquelle la méthode s'appuie. Certains cours portent \
spécifiquement sur la sphère gynécologique, la périnatalité et la \
pédiatrie : organes génitaux, maternité, grossesse, accouchement, \
post-partum, prise en charge du nouveau-né, du nourrisson et de l'enfant \
(développement psychomoteur, réflexes archaïques, troubles de l'oralité). \
Attends-toi à un vocabulaire anatomique, ostéopathique, \
gynéco-obstétrical, pédiatrique et énergétique précis, que Whisper a \
souvent mal reconnu ou remplacé par des mots courants phonétiquement \
proches.
Exemples de termes du domaine : {glossary}

Consignes :
- Corrige UNIQUEMENT les erreurs de transcription : homophonies (par \
exemple « brouillon de culture » → « bouillon de culture »), termes \
anatomiques ou techniques mal reconnus, accords manifestement erronés, \
ponctuation et majuscules.
- Ne reformule pas, ne résume pas, n'omets rien : conserve mot pour mot \
tout ce qui est correct, le découpage en paragraphes et les éventuels \
horodatages [h:mm:ss] en début de paragraphe.
- Si un passage est ambigu et que tu n'es pas raisonnablement sûr de la \
correction, laisse-le tel quel.
- Réponds avec le texte corrigé uniquement, sans commentaire, préambule ni \
mise en forme ajoutée.

Texte à corriger :

"""

# Chunks envoyés à Claude (découpés sur les paragraphes)
CORRECTION_CHUNK_CHARS = 6000


def load_glossary() -> str:
    terms = []
    for path in (SCRIPT_GLOSSARY_FILE, GLOSSARY_FILE):
        if path.exists():
            terms += [line.strip() for line in
                      path.read_text(encoding="utf-8").splitlines()
                      if line.strip() and not line.lstrip().startswith("#")]
    return ", ".join(terms)


def claude_cli_version() -> str | None:
    import shutil
    import subprocess

    if not shutil.which("claude"):
        return None
    try:
        result = subprocess.run(["claude", "--version"],
                                capture_output=True, timeout=30,
                                encoding="utf-8", errors="replace")
        return result.stdout.strip() or "installé"
    except Exception:
        return None


def split_into_chunks(text: str, max_chars: int = CORRECTION_CHUNK_CHARS):
    paragraphs = text.split("\n\n")
    chunks, current, size = [], [], 0
    for paragraph in paragraphs:
        if current and size + len(paragraph) > max_chars:
            chunks.append("\n\n".join(current))
            current, size = [], 0
        current.append(paragraph)
        size += len(paragraph) + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def correct_text(text: str, claude_model: str | None = None) -> str:
    """Corrige le transcript par morceaux via `claude -p` (headless)."""
    import subprocess

    prompt_header = CORRECTION_PROMPT.format(glossary=load_glossary())
    chunks = split_into_chunks(text)
    corrected = []
    for index, chunk in enumerate(chunks, start=1):
        print(f"\r  correction Claude {index}/{len(chunks)}…",
              end="", flush=True)
        command = ["claude", "-p", "--output-format", "text"]
        if claude_model:
            command += ["--model", claude_model]
        # encoding explicite : sous Windows, text=True utiliserait cp1252,
        # incapable d'encoder « → » (présent dans le prompt) ou « œ »
        result = subprocess.run(
            command,
            input=prompt_header + chunk,
            capture_output=True, timeout=900,
            encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"claude -p a échoué (chunk {index}/{len(chunks)}): "
                f"{result.stderr.strip()[:500]}"
            )
        output = result.stdout.strip()
        # garde-fous : si la réponse s'écarte trop de l'original (résumé,
        # refus, commentaire) ou si les horodatages [h:mm:ss] ne sont pas
        # conservés à l'identique, on garde le texte d'origine
        timestamps = re.findall(r"\[\d+:\d{2}:\d{2}\]", chunk)
        if not output or not 0.5 < len(output) / max(len(chunk), 1) < 1.5:
            print(f"\n  chunk {index}: réponse inattendue, texte original "
                  "conservé", file=sys.stderr)
            output = chunk
        elif re.findall(r"\[\d+:\d{2}:\d{2}\]", output) != timestamps:
            print(f"\n  chunk {index}: horodatages modifiés, texte original "
                  "conservé", file=sys.stderr)
            output = chunk
        corrected.append(output)
    print()
    return "\n\n".join(corrected) + "\n"


def setup_windows_cuda_dlls():
    """Rend les DLL cuBLAS/cuDNN des wheels pip visibles pour ctranslate2."""
    if sys.platform != "win32":
        return
    import site

    site_dirs = site.getsitepackages()
    try:
        site_dirs.append(site.getusersitepackages())
    except Exception:
        pass
    for site_dir in site_dirs:
        nvidia_dir = Path(site_dir) / "nvidia"
        if not nvidia_dir.is_dir():
            continue
        for bin_dir in nvidia_dir.glob("*/bin"):
            os.add_dll_directory(str(bin_dir))


def model_cache_info(model_name: str) -> tuple[str, Path]:
    """Retourne (repo Hugging Face, répertoire de cache) du modèle."""
    try:
        from faster_whisper.utils import _MODELS
        repo = _MODELS.get(model_name)
    except Exception:
        repo = None
    repo = repo or f"Systran/faster-whisper-{model_name}"
    hf_home = Path(os.environ.get("HF_HOME",
                                  Path.home() / ".cache" / "huggingface"))
    cache = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", hf_home / "hub"))
    return repo, cache / ("models--" + repo.replace("/", "--"))


def model_is_cached(model_name: str) -> bool:
    """Vérifie si le modèle Whisper est déjà COMPLET dans le cache HF.

    Ne pas se fier au simple contenu du répertoire : un téléchargement
    interrompu y laisse déjà les petits fichiers (config, tokenizer).
    model.bin — les poids, l'essentiel du volume — doit être présent et
    résolu (un lien vers un blob incomplet ne compte pas).
    """
    snapshots = model_cache_info(model_name)[1] / "snapshots"
    if not snapshots.is_dir():
        return False
    return any((snap / "model.bin").exists() for snap in snapshots.iterdir())


def model_blobs_size(model_name: str) -> int:
    """Octets déjà téléchargés (blobs/ uniquement : sous Windows, sans liens
    symboliques, les fichiers sont dupliqués dans snapshots/, ce qui
    fausserait la mesure)."""
    cache_dir = model_cache_info(model_name)[1]
    target = cache_dir / "blobs"
    if not target.is_dir():
        target = cache_dir
    try:
        return sum(f.stat().st_size for f in target.rglob("*")
                   if f.is_file() and not f.is_symlink())
    except OSError:
        return 0


def ensure_model_downloaded(model_name: str):
    """Télécharge le modèle si absent, avec notre propre progression.

    Les barres tqdm de huggingface_hub ne s'affichent pas de façon fiable ;
    on suit à la place la taille du répertoire de cache pendant le
    téléchargement, rapportée à la taille totale annoncée par l'API HF.
    """
    if model_is_cached(model_name):
        return
    import threading

    from faster_whisper import download_model
    from huggingface_hub.utils import disable_progress_bars

    repo = model_cache_info(model_name)[0]
    total = None
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo, files_metadata=True)
        total = sum(s.size or 0 for s in info.siblings) or None
    except Exception:
        pass

    def cache_size() -> int:
        return model_blobs_size(model_name)

    stop = threading.Event()

    def human(size: int) -> str:
        return (f"{size / 1e9:.1f} Go" if size >= 1e9
                else f"{size / 1e6:.0f} Mo")

    def watch():
        last_size = 0
        last_time = time.monotonic()
        final_start = None
        while not stop.wait(1.0):
            size = cache_size()
            now = time.monotonic()
            speed = max(0.0, size - last_size) / max(now - last_time, 1e-6)
            last_size, last_time = size, now
            if total and size >= total:
                # tout est téléchargé, la bibliothèque finalise (copie des
                # fichiers dans snapshots/ sous Windows — ~3 Go, peut
                # prendre une minute sur un disque lent)
                if final_start is None:
                    final_start = now
                print(f"\r  téléchargement du modèle {model_name} 100% "
                      f"({human(total)}) — finalisation… "
                      f"{int(now - final_start)}s ",
                      end="", flush=True)
            elif total:
                print(f"\r  téléchargement du modèle {model_name} "
                      f"{int(size / total * 100):3d}% "
                      f"({human(size)}/{human(total)}, "
                      f"{speed / 1e6:.1f} Mo/s)",
                      end="", flush=True)
            else:
                print(f"\r  téléchargement du modèle {model_name}… "
                      f"{human(size)} ({speed / 1e6:.1f} Mo/s)",
                      end="", flush=True)

    disable_progress_bars()
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        download_model(model_name)
    finally:
        stop.set()
        watcher.join()
        print()


def preflight(model_name: str, need_drive: bool,
              need_claude: bool = False, need_whisper: bool = True) -> bool:
    """Affiche l'état de l'environnement. Retourne False si un ✗ bloque."""
    ok = True

    def check(status: str, label: str):
        # status: "ok" (✓), "info" (–), "fail" (✗)
        nonlocal ok
        if status == "fail":
            ok = False
        mark = {"ok": "✓", "info": "–", "fail": "✗"}[status]
        print(f"  {mark} {label}")

    print("Vérification de l'environnement :")
    check("ok", f"Python {sys.version.split()[0]}")

    if need_whisper:
        setup_windows_cuda_dlls()
        try:
            import av
            import ctranslate2
            import faster_whisper

            check("ok", f"faster-whisper {faster_whisper.__version__} "
                        f"(ctranslate2 {ctranslate2.__version__})")
            check("ok", f"décodage audio/vidéo intégré (PyAV {av.__version__},"
                        " ffmpeg non requis)")
            gpus = ctranslate2.get_cuda_device_count()
            if gpus:
                check("ok", f"GPU CUDA détecté ({gpus} périphérique(s))")
            else:
                check("info",
                      "pas de GPU CUDA → transcription sur CPU (lente)")
        except Exception as exc:
            check("fail", f"bibliothèques de transcription : {exc}")

        if model_is_cached(model_name):
            check("ok", f"modèle {model_name} présent dans le cache")
        else:
            partial = model_blobs_size(model_name)
            if partial > 1e6:
                check("info", f"modèle {model_name} : cache incomplet "
                              f"({partial / 1e9:.1f} Go déjà téléchargés) → "
                              "complété au lancement")
            else:
                check("info", f"modèle {model_name} absent du cache → "
                              "téléchargé au premier lancement "
                              "(~3 Go pour large-v3)")

    if need_claude:
        version = claude_cli_version()
        if version:
            check("ok", f"Claude Code ({version}) pour la correction")
        else:
            check("fail", "Claude Code introuvable (commande `claude`) — "
                          "requis pour --correct/--fix ; voir README.md")
        if SCRIPT_GLOSSARY_FILE.exists():
            count = len([line for line in SCRIPT_GLOSSARY_FILE.read_text(
                encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")])
            check("ok", f"glossaire du domaine ({SCRIPT_GLOSSARY_FILE.name}, "
                        f"{count} termes)")
        else:
            check("fail", f"glossaire du domaine absent "
                          f"({SCRIPT_GLOSSARY_FILE}) — récupérer "
                          "glossaire.txt avec le script")
        if GLOSSARY_FILE.exists():
            check("ok", f"glossaire personnalisé ({GLOSSARY_FILE})")
        else:
            check("info", f"pas de glossaire personnalisé ({GLOSSARY_FILE})")

    if need_drive:
        if CREDENTIALS_FILE.exists():
            check("ok", f"client OAuth Google ({CREDENTIALS_FILE})")
        else:
            check("fail", f"client OAuth Google absent : {CREDENTIALS_FILE} "
                          "(voir README.md, section Google Cloud)")
        if TOKEN_FILE.exists():
            check("ok", "accès Drive déjà autorisé (jeton en cache)")
        else:
            check("info", "accès Drive pas encore autorisé → le navigateur "
                          "s'ouvrira au lancement")

    print()
    return ok


# ---------------------------------------------------------------------------
# Google Drive
# ---------------------------------------------------------------------------


def get_drive_service():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception:
            creds = None
    if not creds or not creds.valid:
        if not CREDENTIALS_FILE.exists():
            sys.exit(
                f"Fichier {CREDENTIALS_FILE} introuvable.\n"
                "Créez un client OAuth « Desktop app » sur "
                "https://console.cloud.google.com/apis/credentials et placez "
                "le JSON téléchargé à cet emplacement (voir README.md)."
            )
        flow = InstalledAppFlow.from_client_secrets_file(
            str(CREDENTIALS_FILE), SCOPES
        )
        creds = flow.run_local_server(port=0)
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(creds.to_json())
    return build("drive", "v3", credentials=creds)


class DriveSession:
    """Appels Drive avec reprises sur connexion neuve.

    Google ferme les connexions restées inactives pendant une longue
    transcription : l'appel suivant échoue alors sur le socket mort
    (ConnectionResetError / WinError 10054). Chaque appel est rejoué avec
    backoff, en reconstruisant le client entre deux tentatives.
    """

    ATTEMPTS = 4

    def __init__(self):
        self._service = None

    @property
    def service(self):
        if self._service is None:
            self._service = get_drive_service()
        return self._service

    def call(self, label: str, fn):
        import httplib2
        from googleapiclient.errors import HttpError

        for attempt in range(1, self.ATTEMPTS + 1):
            try:
                return fn(self.service)
            except Exception as exc:
                retryable = (
                    isinstance(exc, (OSError, httplib2.HttpLib2Error))
                    or (isinstance(exc, HttpError)
                        and exc.resp.status in (429, 500, 502, 503, 504))
                )
                if not retryable or attempt == self.ATTEMPTS:
                    raise
                delay = 2 ** (attempt - 1)
                print(f"\n  {label} : {exc} → nouvelle tentative "
                      f"{attempt}/{self.ATTEMPTS - 1} dans {delay}s…",
                      file=sys.stderr)
                time.sleep(delay)
                self._service = None  # connexion neuve


def extract_folder_id(arg: str) -> str:
    """Accepte l'URL complète d'un dossier Drive ou son ID nu."""
    match = re.search(r"/folders/([A-Za-z0-9_-]+)", arg)
    if match:
        return match.group(1)
    return arg.split("?")[0]


FOLDER_MIME = "application/vnd.google-apps.folder"


SHORTCUT_MIME = "application/vnd.google-apps.shortcut"


def scan_drive_folder(drive, folder_id: str,
                      debug: bool = False) -> tuple[list[dict], set]:
    """Parcourt le dossier ET ses sous-dossiers.

    Retourne (fichiers audio/vidéo, Docs existants). Chaque fichier porte
    en plus `parentId` (le sous-dossier qui le contient, où créer le Doc)
    et `relpath` (chemin relatif affiché). Les Docs existants — collectés
    pendant le même parcours, sans appels supplémentaires — forment un
    ensemble de couples (parentId, nom) utilisé pour ignorer les fichiers
    déjà transcrits.

    Les raccourcis Drive (dossiers/fichiers « ajoutés depuis un partage »,
    mimeType shortcut) sont résolus vers leur cible.
    """
    files = []
    existing_docs = {}  # (parentId, nom) -> id du Doc
    queue = [(folder_id, "")]
    visited = {folder_id}
    while queue:
        current_id, prefix = queue.pop(0)
        query = (
            f"'{current_id}' in parents and trashed = false and "
            "(mimeType contains 'audio/' or mimeType contains 'video/' "
            f"or mimeType = '{FOLDER_MIME}' "
            f"or mimeType = '{GOOGLE_DOC_MIME}' "
            f"or mimeType = '{SHORTCUT_MIME}')"
        )
        page_token = None
        count = 0

        def enqueue_folder(sub_id: str, name: str):
            if sub_id not in visited:
                visited.add(sub_id)
                queue.append((sub_id, f"{prefix}{name}/"))

        while True:
            response = (
                drive.files()
                .list(
                    q=query,
                    fields="nextPageToken, files(id, name, mimeType, size, "
                           "shortcutDetails, appProperties)",
                    orderBy="name",
                    pageSize=100,
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            for f in response.get("files", []):
                count += 1
                mime = f["mimeType"]
                if debug:
                    print(f"  [debug] {prefix or './'} : {f['name']} "
                          f"({mime})")
                if mime == FOLDER_MIME:
                    enqueue_folder(f["id"], f["name"])
                elif mime == GOOGLE_DOC_MIME:
                    existing_docs[(current_id, f["name"])] = {
                        "id": f["id"],
                        "corrected": (f.get("appProperties") or {})
                        .get(CORRECTED_PROP) == "1",
                    }
                elif mime == SHORTCUT_MIME:
                    details = f.get("shortcutDetails") or {}
                    target_id = details.get("targetId")
                    target_mime = details.get("targetMimeType", "")
                    if not target_id:
                        continue
                    if target_mime == FOLDER_MIME:
                        enqueue_folder(target_id, f["name"])
                    elif target_mime.startswith(("audio/", "video/")):
                        files.append({
                            "id": target_id,
                            "name": f["name"],
                            "mimeType": target_mime,
                            "size": 0,
                            "parentId": current_id,
                            "relpath": prefix + f["name"],
                        })
                else:
                    f["parentId"] = current_id
                    f["relpath"] = prefix + f["name"]
                    files.append(f)
            page_token = response.get("nextPageToken")
            if not page_token:
                break
        if debug:
            print(f"  [debug] {prefix or './'} → {count} élément(s), "
                  f"{len(queue)} dossier(s) en attente")
    files.sort(key=lambda f: f["relpath"])
    return files, existing_docs


def download_file(drive, file: dict, dest_dir: Path) -> Path:
    from googleapiclient.http import MediaIoBaseDownload

    size_mb = int(file.get("size", 0)) / 1e6
    print(f"  téléchargement de {size_mb:.0f} Mo…", end="", flush=True)
    dest = dest_dir / file["name"]
    request = drive.files().get_media(fileId=file["id"], supportsAllDrives=True)
    start = time.monotonic()
    with open(dest, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=16 * 1024 * 1024)
        done = False
        while not done:
            status, done = downloader.next_chunk(num_retries=3)
            if status:
                done_mb = status.resumable_progress / 1e6
                elapsed = time.monotonic() - start
                speed = done_mb / elapsed if elapsed > 0 else 0
                print(f"\r  téléchargement {int(status.progress() * 100):3d}% "
                      f"({done_mb:.0f}/{size_mb:.0f} Mo, {speed:.1f} Mo/s)",
                      end="", flush=True)
    print()
    return dest


def create_google_doc(drive, folder_id: str, doc_name: str, text: str,
                      corrected: bool = False) -> str:
    from googleapiclient.http import MediaIoBaseUpload

    media = MediaIoBaseUpload(
        io.BytesIO(text.encode("utf-8")),
        mimetype="text/plain",
        resumable=True,
    )
    metadata = {
        "name": doc_name,
        "mimeType": GOOGLE_DOC_MIME,
        "parents": [folder_id],
    }
    if corrected:
        metadata["appProperties"] = {CORRECTED_PROP: "1"}
    doc = (
        drive.files()
        .create(body=metadata, media_body=media, fields="id, webViewLink",
                supportsAllDrives=True)
        .execute()
    )
    return doc.get("webViewLink", doc["id"])


def export_google_doc(drive, doc_id: str) -> str:
    """Récupère le contenu texte d'un Google Doc."""
    data = (
        drive.files()
        .export(fileId=doc_id, mimeType="text/plain")
        .execute()
    )
    return data.decode("utf-8").lstrip("﻿")


def update_google_doc(drive, doc_id: str, text: str):
    """Remplace le contenu d'un Google Doc (même id, même lien) et le
    marque comme corrigé (appProperties)."""
    from googleapiclient.http import MediaIoBaseUpload

    media = MediaIoBaseUpload(
        io.BytesIO(text.encode("utf-8")),
        mimetype="text/plain",
        resumable=True,
    )
    (
        drive.files()
        .update(fileId=doc_id, media_body=media,
                body={"appProperties": {CORRECTED_PROP: "1"}},
                supportsAllDrives=True)
        .execute()
    )


# ---------------------------------------------------------------------------
# Whisper
# ---------------------------------------------------------------------------


def load_model(model_name: str):
    """Charge le modèle sur GPU si possible, sinon CPU (int8)."""
    setup_windows_cuda_dlls()
    import ctranslate2
    from faster_whisper import WhisperModel

    ensure_model_downloaded(model_name)
    print(f"Chargement du modèle {model_name}…", flush=True)
    attempts = []
    if ctranslate2.get_cuda_device_count() > 0:
        attempts += [("cuda", "float16"), ("cuda", "int8_float16")]
    attempts.append(("cpu", "int8"))

    last_error = None
    for device, compute_type in attempts:
        try:
            start = time.monotonic()
            model = WhisperModel(model_name, device=device,
                                 compute_type=compute_type)
            print(f"Modèle {model_name} chargé sur {device} ({compute_type}) "
                  f"en {time.monotonic() - start:.0f}s")
            return model
        except Exception as exc:  # OOM GPU, DLL manquante, etc.
            print(f"  échec sur {device}/{compute_type}: {exc}",
                  file=sys.stderr)
            last_error = exc
    raise last_error


def format_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def transcribe_file(model, path: Path, language: str | None,
                    timestamps: bool) -> str:
    # transcribe() décode toute la piste audio et exécute le VAD avant de
    # produire le premier segment — long pour une vidéo, d'où le message.
    print("  extraction de l'audio et détection de la parole (VAD)…",
          end="", flush=True)
    prep_start = time.monotonic()
    segments, info = model.transcribe(
        str(path),
        language=language,
        vad_filter=True,
    )
    print(f" {time.monotonic() - prep_start:.0f}s")
    duration = info.duration or 0
    print(f"  durée {format_timestamp(duration)}, "
          f"langue détectée: {info.language} "
          f"(p={info.language_probability:.2f})")

    paragraphs: list[str] = []
    current: list[str] = []
    current_start = 0.0
    previous_end = 0.0
    start_time = time.monotonic()

    def flush():
        if not current:
            return
        text = " ".join(current)
        if timestamps:
            text = f"[{format_timestamp(current_start)}] {text}"
        paragraphs.append(text)
        current.clear()

    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        gap = segment.start - previous_end
        too_long = sum(len(part) for part in current) > PARAGRAPH_MAX_CHARS
        if current and (gap >= PARAGRAPH_GAP_SECONDS or too_long):
            flush()
        if not current:
            current_start = segment.start
        current.append(text)
        previous_end = segment.end
        if duration:
            percent = min(100, int(segment.end / duration * 100))
            elapsed = time.monotonic() - start_time
            speed = segment.end / elapsed if elapsed > 0 else 0
            print(f"\r  transcription {percent:3d}% "
                  f"({format_timestamp(segment.end)}, {speed:.1f}x temps réel)",
                  end="", flush=True)
    flush()
    print()
    return "\n\n".join(paragraphs) + "\n"


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


def run_drive(args) -> int:
    session = DriveSession()
    folder_id = extract_folder_id(args.source)
    files, existing_docs = session.call(
        "listing du dossier",
        lambda d: scan_drive_folder(d, folder_id, debug=args.debug))
    if not files:
        print("Aucun fichier audio/vidéo dans ce dossier "
              "(sous-dossiers compris).")
        return 0
    print(f"{len(files)} fichier(s) audio/vidéo trouvé(s) "
          "(sous-dossiers compris).")

    model = None
    done, skipped, failed = [], [], []
    for file in files:
        base_name = Path(file["name"]).stem
        print(f"\n=== {file['relpath']} ===")
        try:
            if args.txt:
                out_path = (Path(args.txt)
                            / Path(file["relpath"]).with_suffix(".txt"))
                if out_path.exists() and not args.force:
                    print("  déjà transcrit (fichier .txt existant), ignoré")
                    skipped.append(file["relpath"])
                    continue
            elif (not args.force
                  and (file["parentId"], base_name) in existing_docs):
                print("  déjà transcrit (Google Doc existant), ignoré")
                skipped.append(file["relpath"])
                continue

            if model is None:
                model = load_model(args.model)

            with tempfile.TemporaryDirectory(prefix="transcribe-") as tmp:
                local = session.call(
                    "téléchargement",
                    lambda d: download_file(d, file, Path(tmp)))
                text = transcribe_file(model, local, args.language,
                                       args.timestamps)
            if args.correct:
                text = correct_text(text, args.claude_model)

            if args.txt:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(text, encoding="utf-8")
                print(f"  écrit: {out_path}")
            else:
                link = session.call(
                    "création du Doc",
                    lambda d: create_google_doc(d, file["parentId"],
                                                base_name, text,
                                                corrected=args.correct))
                print(f"  Google Doc créé: {link}")
            done.append(file["relpath"])
        except Exception as exc:
            print(f"  ERREUR: {exc}", file=sys.stderr)
            failed.append(file["relpath"])

    print(f"\nTerminé: {len(done)} transcrit(s), {len(skipped)} ignoré(s), "
          f"{len(failed)} en erreur.")
    for name in failed:
        print(f"  échec: {name}", file=sys.stderr)
    return 1 if failed else 0


def run_local(args, source: Path) -> int:
    if source.is_dir():
        files = sorted(
            p for p in source.rglob("*")
            if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS
        )
    else:
        files = [source]
    if not files:
        print("Aucun fichier audio/vidéo dans ce dossier "
              "(sous-dossiers compris).")
        return 0
    print(f"{len(files)} fichier(s) audio/vidéo trouvé(s) "
          "(sous-dossiers compris).")

    out_dir = Path(args.txt) if args.txt else None
    model = None
    done, skipped, failed = [], [], []
    for path in files:
        rel = (path.relative_to(source) if source.is_dir()
               else Path(path.name))
        # avec --txt, l'arborescence des sous-dossiers est reproduite
        out_path = (out_dir / rel.with_suffix(".txt") if out_dir
                    else path.with_suffix(".txt"))
        print(f"\n=== {rel} ===")
        try:
            if out_path.exists() and not args.force:
                print("  déjà transcrit (fichier .txt existant), ignoré")
                skipped.append(str(rel))
                continue
            if model is None:
                model = load_model(args.model)
            text = transcribe_file(model, path, args.language, args.timestamps)
            if args.correct:
                text = correct_text(text, args.claude_model)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text, encoding="utf-8")
            if args.correct:
                # marque le .txt comme déjà corrigé pour un futur --fix
                root = out_dir or (source if source.is_dir() else path.parent)
                mark_corrected(root, str(out_path.relative_to(root)),
                               load_corrected_state(root))
            print(f"  écrit: {out_path}")
            done.append(str(rel))
        except Exception as exc:
            print(f"  ERREUR: {exc}", file=sys.stderr)
            failed.append(str(rel))

    print(f"\nTerminé: {len(done)} transcrit(s), {len(skipped)} ignoré(s), "
          f"{len(failed)} en erreur.")
    return 1 if failed else 0


def run_fix_drive(args) -> int:
    """Corrige via Claude les Google Docs déjà produits (sans retranscrire).

    Seuls les Docs dont le nom correspond à un fichier audio/vidéo du même
    sous-dossier sont traités ; le Doc est mis à jour en place (même lien).
    """
    session = DriveSession()
    folder_id = extract_folder_id(args.source)
    files, existing_docs = session.call(
        "listing du dossier",
        lambda d: scan_drive_folder(d, folder_id, debug=args.debug))

    targets = []
    skipped = []
    for file in files:
        base_name = Path(file["name"]).stem
        doc = existing_docs.get((file["parentId"], base_name))
        if not doc:
            continue
        if doc["corrected"] and not args.force:
            skipped.append(file["relpath"])
        else:
            targets.append((file["relpath"], base_name, doc["id"]))
    if skipped:
        print(f"{len(skipped)} transcript(s) déjà corrigé(s), ignoré(s) "
              "(--force pour recorriger).")
    if not targets:
        print("Aucun Google Doc de transcript à corriger dans ce dossier.")
        return 0
    print(f"{len(targets)} transcript(s) à corriger "
          "(sous-dossiers compris).")

    done, failed = [], []
    for relpath, base_name, doc_id in targets:
        print(f"\n=== {relpath} ===")
        try:
            text = session.call(
                "lecture du Doc",
                lambda d: export_google_doc(d, doc_id))
            corrected = correct_text(text, args.claude_model)
            session.call(
                "mise à jour du Doc",
                lambda d: update_google_doc(d, doc_id, corrected))
            print(f"  Doc « {base_name} » corrigé et mis à jour")
            done.append(relpath)
        except Exception as exc:
            print(f"  ERREUR: {exc}", file=sys.stderr)
            failed.append(relpath)

    print(f"\nTerminé: {len(done)} corrigé(s), {len(skipped)} ignoré(s), "
          f"{len(failed)} en erreur.")
    for name in failed:
        print(f"  échec: {name}", file=sys.stderr)
    return 1 if failed else 0


def local_state_file(source: Path) -> Path:
    root = source if source.is_dir() else source.parent
    return root / ".transcribe-state.json"


def load_corrected_state(source: Path) -> set:
    import json

    state_path = local_state_file(source)
    if state_path.exists():
        try:
            return set(json.loads(
                state_path.read_text(encoding="utf-8")).get("corrected", []))
        except Exception:
            pass
    return set()


def mark_corrected(source: Path, rel: str, corrected: set):
    import json

    corrected.add(rel)
    local_state_file(source).write_text(
        json.dumps({"corrected": sorted(corrected)},
                   ensure_ascii=False, indent=1),
        encoding="utf-8")


def run_fix_local(args, source: Path) -> int:
    """Corrige via Claude des fichiers .txt existants, en place."""
    if source.is_dir():
        files = sorted(p for p in source.rglob("*.txt") if p.is_file())
    elif source.suffix.lower() == ".txt":
        files = [source]
    else:
        print("--fix en local attend un fichier .txt ou un dossier "
              "en contenant.", file=sys.stderr)
        return 1
    if not files:
        print("Aucun fichier .txt à corriger.")
        return 0

    corrected_state = load_corrected_state(source)
    done, skipped, failed = [], [], []
    todo = []
    for path in files:
        rel = str(path.relative_to(source) if source.is_dir() else path.name)
        if rel in corrected_state and not args.force:
            skipped.append(rel)
        else:
            todo.append((path, rel))
    if skipped:
        print(f"{len(skipped)} fichier(s) déjà corrigé(s), ignoré(s) "
              "(--force pour recorriger).")
    print(f"{len(todo)} fichier(s) .txt à corriger.")

    for path, rel in todo:
        print(f"\n=== {rel} ===")
        try:
            corrected = correct_text(path.read_text(encoding="utf-8"),
                                     args.claude_model)
            path.write_text(corrected, encoding="utf-8")
            mark_corrected(source, rel, corrected_state)
            print(f"  corrigé: {path}")
            done.append(rel)
        except Exception as exc:
            print(f"  ERREUR: {exc}", file=sys.stderr)
            failed.append(rel)

    print(f"\nTerminé: {len(done)} corrigé(s), {len(skipped)} ignoré(s), "
          f"{len(failed)} en erreur.")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcrit les fichiers audio/vidéo d'un dossier Google "
                    "Drive (ou local) avec Whisper, et crée un Google Doc "
                    "par fichier (ou un .txt).",
    )
    parser.add_argument(
        "source",
        nargs="?",
        help="URL ou ID d'un dossier Google Drive, ou chemin local "
             "(fichier ou dossier)",
    )
    parser.add_argument("--check", action="store_true",
                        help="vérifier l'installation (dépendances, GPU, "
                             "modèle, accès Google) et quitter")
    parser.add_argument("--debug", action="store_true",
                        help="afficher le détail du parcours Drive "
                             "(chaque élément trouvé et son type)")
    parser.add_argument("--language", default=None, metavar="LANG",
                        help="langue de l'audio (fr, en, ...) ; "
                             "auto-détection par défaut")
    parser.add_argument("--model", default="large-v3",
                        help="modèle Whisper (défaut: large-v3 ; "
                             "utiliser tiny pour un test rapide)")
    parser.add_argument("--timestamps", action="store_true",
                        help="préfixer chaque paragraphe de [h:mm:ss]")
    parser.add_argument("--force", action="store_true",
                        help="retraiter même si déjà transcrit (ou déjà "
                             "corrigé, avec --fix)")
    parser.add_argument("--txt", metavar="DIR", default=None,
                        help="écrire des fichiers .txt dans DIR au lieu de "
                             "créer des Google Docs")
    parser.add_argument("--correct", action="store_true",
                        help="corriger le transcript via Claude "
                             "(vocabulaire anatomique, homophonies) avant "
                             "d'écrire le Doc/.txt")
    parser.add_argument("--fix", action="store_true",
                        help="ne rien transcrire : corriger via Claude les "
                             "Google Docs (ou .txt) déjà produits pour les "
                             "fichiers audio/vidéo du dossier")
    parser.add_argument("--claude-model", default=None, metavar="MODEL",
                        help="modèle passé à `claude -p` pour la correction "
                             "(défaut: modèle configuré dans Claude Code)")
    args = parser.parse_args()

    if args.language and args.language.lower() == "auto":
        args.language = None

    if not args.check and not args.source:
        parser.error("préciser une source (dossier Drive ou chemin local), "
                     "ou --check pour vérifier l'installation")

    # Le pipeline tourne dans un thread démon : le thread principal ne fait
    # qu'attendre par petites tranches, et reste donc capable de traiter
    # Ctrl+C immédiatement, même pendant les longs appels natifs
    # (téléchargement Hugging Face, chargement/inférence ctranslate2) qui
    # bloqueraient sinon la livraison de KeyboardInterrupt.
    import threading

    outcome = {}

    def work():
        try:
            outcome["code"] = run(args)
        except SystemExit as exc:
            if isinstance(exc.code, int):
                outcome["code"] = exc.code
            else:
                if exc.code is not None:
                    print(exc.code, file=sys.stderr)
                outcome["code"] = 1
        except Exception:
            import traceback
            traceback.print_exc()
            outcome["code"] = 1

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    try:
        while worker.is_alive():
            worker.join(0.2)
    except KeyboardInterrupt:
        print("\nInterrompu (Ctrl+C).", file=sys.stderr)
        sys.stderr.flush()
        sys.stdout.flush()
        # sortie immédiate : les threads natifs (HF, ctranslate2) ne peuvent
        # pas être arrêtés proprement
        os._exit(130)
    return outcome.get("code", 1)


def run(args) -> int:
    if args.check:
        return 0 if preflight(args.model, need_drive=True,
                              need_claude=True) else 1

    local = Path(args.source).expanduser()
    is_local = local.exists()
    need_claude = args.correct or args.fix
    if not preflight(args.model, need_drive=not is_local,
                     need_claude=need_claude, need_whisper=not args.fix):
        return 1
    if args.fix:
        return run_fix_local(args, local) if is_local else run_fix_drive(args)
    if is_local:
        return run_local(args, local)
    return run_drive(args)


if __name__ == "__main__":
    sys.exit(main())
