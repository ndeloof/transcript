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

CONFIG_DIR = Path.home() / ".transcribe"
CREDENTIALS_FILE = CONFIG_DIR / "credentials.json"
TOKEN_FILE = CONFIG_DIR / "token.json"
SCOPES = ["https://www.googleapis.com/auth/drive"]

GOOGLE_DOC_MIME = "application/vnd.google-apps.document"

# Extensions reconnues en mode local (source = fichier/dossier sur disque)
MEDIA_EXTENSIONS = {
    ".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma", ".aiff",
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg",
}

# Pause entre deux segments au-delà de laquelle on démarre un paragraphe
PARAGRAPH_GAP_SECONDS = 2.0
# Longueur au-delà de laquelle on coupe le paragraphe à la prochaine pause
PARAGRAPH_MAX_CHARS = 1200


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

    repo, cache_dir = model_cache_info(model_name)
    total = None
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo, files_metadata=True)
        total = sum(s.size or 0 for s in info.siblings) or None
    except Exception:
        pass

    def cache_size() -> int:
        # ne mesurer que blobs/ : sous Windows (pas de liens symboliques),
        # chaque fichier est dupliqué dans snapshots/, ce qui ferait
        # afficher 100% à la moitié du téléchargement réel
        target = cache_dir / "blobs"
        if not target.is_dir():
            target = cache_dir
        try:
            return sum(f.stat().st_size for f in target.rglob("*")
                       if f.is_file() and not f.is_symlink())
        except OSError:
            return 0

    stop = threading.Event()

    def human(size: int) -> str:
        return (f"{size / 1e9:.1f} Go" if size >= 1e9
                else f"{size / 1e6:.0f} Mo")

    def watch():
        last_size = 0
        last_time = time.monotonic()
        while not stop.wait(1.0):
            size = cache_size()
            now = time.monotonic()
            speed = max(0.0, size - last_size) / max(now - last_time, 1e-6)
            last_size, last_time = size, now
            if total and size >= total:
                # tout est téléchargé, la bibliothèque finalise (copie des
                # fichiers dans snapshots/ sous Windows)
                print(f"\r  téléchargement du modèle {model_name} 100% "
                      f"({human(total)}) — finalisation…    ",
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


def preflight(model_name: str, need_drive: bool) -> bool:
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

    setup_windows_cuda_dlls()
    try:
        import av
        import ctranslate2
        import faster_whisper

        check("ok", f"faster-whisper {faster_whisper.__version__} "
                    f"(ctranslate2 {ctranslate2.__version__})")
        check("ok", f"décodage audio/vidéo intégré (PyAV {av.__version__}, "
                    "ffmpeg non requis)")
        gpus = ctranslate2.get_cuda_device_count()
        if gpus:
            check("ok", f"GPU CUDA détecté ({gpus} périphérique(s))")
        else:
            check("info", "pas de GPU CUDA → transcription sur CPU (lente)")
    except Exception as exc:
        check("fail", f"bibliothèques de transcription : {exc}")

    if model_is_cached(model_name):
        check("ok", f"modèle {model_name} présent dans le cache")
    else:
        check("info", f"modèle {model_name} absent du cache → "
                      "téléchargé au premier lancement (~3 Go pour large-v3)")

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


def extract_folder_id(arg: str) -> str:
    """Accepte l'URL complète d'un dossier Drive ou son ID nu."""
    match = re.search(r"/folders/([A-Za-z0-9_-]+)", arg)
    if match:
        return match.group(1)
    return arg.split("?")[0]


def drive_quote(name: str) -> str:
    return name.replace("\\", "\\\\").replace("'", "\\'")


def list_media_files(drive, folder_id: str) -> list[dict]:
    query = (
        f"'{folder_id}' in parents and trashed = false and "
        "(mimeType contains 'audio/' or mimeType contains 'video/')"
    )
    files = []
    page_token = None
    while True:
        response = (
            drive.files()
            .list(
                q=query,
                fields="nextPageToken, files(id, name, mimeType, size)",
                orderBy="name",
                pageSize=100,
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return files


def find_existing_doc(drive, folder_id: str, doc_name: str) -> bool:
    query = (
        f"'{folder_id}' in parents and trashed = false and "
        f"mimeType = '{GOOGLE_DOC_MIME}' and name = '{drive_quote(doc_name)}'"
    )
    response = (
        drive.files()
        .list(q=query, fields="files(id)", pageSize=1,
              supportsAllDrives=True, includeItemsFromAllDrives=True)
        .execute()
    )
    return bool(response.get("files"))


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
            status, done = downloader.next_chunk()
            if status:
                done_mb = status.resumable_progress / 1e6
                elapsed = time.monotonic() - start
                speed = done_mb / elapsed if elapsed > 0 else 0
                print(f"\r  téléchargement {int(status.progress() * 100):3d}% "
                      f"({done_mb:.0f}/{size_mb:.0f} Mo, {speed:.1f} Mo/s)",
                      end="", flush=True)
    print()
    return dest


def create_google_doc(drive, folder_id: str, doc_name: str, text: str) -> str:
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
    doc = (
        drive.files()
        .create(body=metadata, media_body=media, fields="id, webViewLink",
                supportsAllDrives=True)
        .execute()
    )
    return doc.get("webViewLink", doc["id"])


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
    drive = get_drive_service()
    folder_id = extract_folder_id(args.source)
    files = list_media_files(drive, folder_id)
    if not files:
        print("Aucun fichier audio/vidéo dans ce dossier.")
        return 0
    print(f"{len(files)} fichier(s) audio/vidéo trouvé(s).")

    model = None
    done, skipped, failed = [], [], []
    for file in files:
        base_name = Path(file["name"]).stem
        print(f"\n=== {file['name']} ===")
        try:
            if args.txt:
                out_path = Path(args.txt) / f"{base_name}.txt"
                if out_path.exists() and not args.force:
                    print("  déjà transcrit (fichier .txt existant), ignoré")
                    skipped.append(file["name"])
                    continue
            elif not args.force and find_existing_doc(drive, folder_id, base_name):
                print("  déjà transcrit (Google Doc existant), ignoré")
                skipped.append(file["name"])
                continue

            if model is None:
                model = load_model(args.model)

            with tempfile.TemporaryDirectory(prefix="transcribe-") as tmp:
                local = download_file(drive, file, Path(tmp))
                text = transcribe_file(model, local, args.language,
                                       args.timestamps)

            if args.txt:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(text, encoding="utf-8")
                print(f"  écrit: {out_path}")
            else:
                link = create_google_doc(drive, folder_id, base_name, text)
                print(f"  Google Doc créé: {link}")
            done.append(file["name"])
        except Exception as exc:
            print(f"  ERREUR: {exc}", file=sys.stderr)
            failed.append(file["name"])

    print(f"\nTerminé: {len(done)} transcrit(s), {len(skipped)} ignoré(s), "
          f"{len(failed)} en erreur.")
    for name in failed:
        print(f"  échec: {name}", file=sys.stderr)
    return 1 if failed else 0


def run_local(args, source: Path) -> int:
    if source.is_dir():
        files = sorted(
            p for p in source.iterdir()
            if p.suffix.lower() in MEDIA_EXTENSIONS
        )
    else:
        files = [source]
    if not files:
        print("Aucun fichier audio/vidéo dans ce dossier.")
        return 0
    print(f"{len(files)} fichier(s) audio/vidéo trouvé(s).")

    out_dir = Path(args.txt) if args.txt else None
    model = None
    done, skipped, failed = [], [], []
    for path in files:
        out_path = (out_dir or path.parent) / f"{path.stem}.txt"
        print(f"\n=== {path.name} ===")
        try:
            if out_path.exists() and not args.force:
                print("  déjà transcrit (fichier .txt existant), ignoré")
                skipped.append(path.name)
                continue
            if model is None:
                model = load_model(args.model)
            text = transcribe_file(model, path, args.language, args.timestamps)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text, encoding="utf-8")
            print(f"  écrit: {out_path}")
            done.append(path.name)
        except Exception as exc:
            print(f"  ERREUR: {exc}", file=sys.stderr)
            failed.append(path.name)

    print(f"\nTerminé: {len(done)} transcrit(s), {len(skipped)} ignoré(s), "
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
    parser.add_argument("--language", default=None, metavar="LANG",
                        help="langue de l'audio (fr, en, ...) ; "
                             "auto-détection par défaut")
    parser.add_argument("--model", default="large-v3",
                        help="modèle Whisper (défaut: large-v3 ; "
                             "utiliser tiny pour un test rapide)")
    parser.add_argument("--timestamps", action="store_true",
                        help="préfixer chaque paragraphe de [h:mm:ss]")
    parser.add_argument("--force", action="store_true",
                        help="retranscrire même si la sortie existe déjà")
    parser.add_argument("--txt", metavar="DIR", default=None,
                        help="écrire des fichiers .txt dans DIR au lieu de "
                             "créer des Google Docs")
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
        return 0 if preflight(args.model, need_drive=True) else 1

    local = Path(args.source).expanduser()
    is_local = local.exists()
    if not preflight(args.model, need_drive=not is_local):
        return 1
    if is_local:
        return run_local(args, local)
    return run_drive(args)


if __name__ == "__main__":
    sys.exit(main())
