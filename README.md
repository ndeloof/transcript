# transcribe

Transcrit des fichiers audio/vidéo stockés sur Google Drive avec
[Whisper](https://github.com/openai/whisper) **large-v3** en local
(via [faster-whisper](https://github.com/SYSTRAN/faster-whisper)),
et crée pour chacun un **Google Doc** contenant le transcript, dans le
même dossier Drive.

Un seul fichier : `transcribe.py`, lancé avec [uv](https://docs.astral.sh/uv/)
qui installe automatiquement Python et les dépendances au premier lancement.

## Pré-requis Google Cloud (une fois, ~5 min)

1. **Créer un projet** :
   <https://console.cloud.google.com/projectcreate>
   (nom libre, ex. `transcribe`). Sélectionner ensuite ce projet dans le
   bandeau en haut de la console pour toutes les étapes suivantes.
2. **Activer l'API Google Drive** :
   <https://console.cloud.google.com/apis/library/drive.googleapis.com>
   → bouton **Enable**.
3. **Configurer l'écran de consentement OAuth** :
   <https://console.cloud.google.com/auth/overview>
   → **Get started** : nom de l'app libre, votre email, audience
   **External**, puis valider.
   Ajouter ensuite votre compte comme **test user** :
   <https://console.cloud.google.com/auth/audience>
   → section « Test users » → **Add users** → `nicolas.deloof@gmail.com`.
4. **Créer le client OAuth** :
   <https://console.cloud.google.com/auth/clients>
   → **Create client** → type **Desktop app**. Télécharger le JSON
   (bouton ⬇) et l'enregistrer sous `~/.transcribe/credentials.json`.

> ⚠️ Tant que l'app OAuth est en mode « Testing », Google fait expirer le
> refresh token au bout de **7 jours** (il faudra ré-autoriser dans le
> navigateur). Pour éviter ça, passer l'app « In production » (l'écran
> « app non vérifiée » s'affiche une fois, sans conséquence pour un usage
> personnel).

## Installation

### Windows (GPU NVIDIA — recommandé, ~10-20x temps réel)

Guide détaillé pas à pas (machine sans outils de dev, Python compris) :
**[INSTALL-WINDOWS.md](INSTALL-WINDOWS.md)**.

En résumé : `winget install astral-sh.uv`, copier `transcribe.py`, c'est
tout — uv installe Python automatiquement, et les DLL CUDA (cuBLAS, cuDNN)
viennent des wheels pip. Seul un **driver NVIDIA récent** est requis (pas
besoin d'installer le CUDA Toolkit).

VRAM : large-v3 nécessite ~5 Go en float16 ; si le GPU en a moins, le script
bascule automatiquement en int8 (~3 Go), puis en CPU en dernier recours.

### macOS

Fonctionne aussi (CPU uniquement, lent avec large-v3). Utile pour tester le
pipeline avec `--model tiny`.

## Usage

```bash
# Tous les fichiers audio/vidéo d'un dossier Drive (sous-dossiers compris)
# → un Google Doc chacun, créé à côté de son fichier source
uv run transcribe.py "https://drive.google.com/drive/folders/<ID>"

# Forcer la langue, ajouter des timestamps [h:mm:ss] par paragraphe
uv run transcribe.py <ID-dossier> --language fr --timestamps

# Sortie en .txt locaux plutôt qu'en Google Docs
uv run transcribe.py <ID-dossier> --txt ./transcripts

# Source locale (fichier ou dossier) → .txt à côté des fichiers
uv run transcribe.py ~/Videos/reunion.mp4

# Test rapide du pipeline complet avec un petit modèle
uv run transcribe.py <ID-dossier> --model tiny

# Vérifier l'installation (dépendances, GPU, modèle, accès Google) sans rien transcrire
uv run transcribe.py --check

# Corriger le transcript via Claude (vocabulaire anatomique, homophonies)
# avant de créer le Doc
uv run transcribe.py <ID-dossier> --correct

# Corriger les Google Docs DÉJÀ produits, sans retranscrire l'audio
uv run transcribe.py <ID-dossier> --fix
```

Le même bilan d'environnement s'affiche au début de chaque lancement.

Au premier lancement : le navigateur s'ouvre pour autoriser l'accès Drive
(jeton mis en cache dans `~/.transcribe/token.json`), et le modèle large-v3
(~3 Go) est téléchargé depuis Hugging Face (mis en cache ensuite).

> Un avertissement « unauthenticated requests to the HF Hub » peut
> s'afficher : il est sans conséquence (modèles publics). Pour de
> meilleures limites de débit lors du premier téléchargement, créer un
> token **Read** sur <https://huggingface.co/settings/tokens> et le
> définir dans la variable d'environnement `HF_TOKEN`
> (PowerShell : `setx HF_TOKEN "hf_xxxx"` ; macOS : `export HF_TOKEN=...`).

Les sous-dossiers sont parcourus récursivement ; chaque Google Doc est créé
dans le sous-dossier de son fichier source (avec `--txt`, l'arborescence est
reproduite). Les fichiers déjà transcrits (un Google Doc du même nom existe
dans le même sous-dossier) sont ignorés et passés sans re-traitement —
`--force` pour retranscrire.

## Correction Claude (`--correct` / `--fix`)

Whisper transcrit phonétiquement et se trompe sur les homophonies
(« bouillon de culture » → « brouillon de culture ») et le vocabulaire
anatomique. La correction passe le transcript dans **Claude** avec un prompt
contenant le contexte technique (thérapie manuelle, ostéopathie, anatomie)
et un glossaire de termes du domaine :

- `--correct` : corrige chaque transcript juste après la transcription,
  avant de créer le Google Doc ;
- `--fix` : **ne retranscrit rien** — corrige les Google Docs (ou .txt)
  déjà produits et les met à jour en place (même lien). Seuls les Docs
  correspondant à un fichier audio/vidéo du même sous-dossier sont traités.

Pré-requis : **Claude Code** installé et connecté (une souscription
Claude Pro/Max suffit, pas besoin de clé API) — la correction appelle
`claude -p` en mode headless. Installation : <https://claude.ai/download>,
puis lancer `claude` une fois pour se connecter.

Le glossaire du domaine vit dans **[`glossaire.txt`](glossaire.txt)**
(versionné à côté du script — somatopathie/méthode Poyet, crânio-sacré,
anatomie, gynécologie, maternité/périnatalité, énergétique) : enrichis-le
directement, un terme par ligne. Des ajouts personnels locaux sont aussi
possibles dans `~/.transcribe/glossaire.txt`. `--claude-model` permet de
choisir le modèle (défaut : celui configuré dans Claude Code).
