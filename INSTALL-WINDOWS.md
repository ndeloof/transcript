# Installation sous Windows (machine non-dev)

Guide pas à pas pour installer tous les pré-requis sur un PC Windows 10/11
avec GPU NVIDIA, sans aucun outil de développement préinstallé.

Toutes les commandes se tapent dans **PowerShell** : menu Démarrer → taper
`powershell` → Entrée (pas besoin de « Exécuter en tant qu'administrateur »).

## 1. Installer uv (gestionnaire Python)

C'est le **seul outil à installer**. Il télécharge automatiquement Python et
toutes les bibliothèques nécessaires au premier lancement du script — il n'y
a donc **pas besoin d'installer Python** séparément.

```powershell
winget install astral-sh.uv
```

Si `winget` n'est pas disponible (Windows 10 ancien), utiliser l'installateur
officiel à la place :

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Puis **fermer et rouvrir PowerShell** (pour que la commande `uv` soit trouvée)
et vérifier :

```powershell
uv --version
```

## 2. Vérifier le driver NVIDIA

Le GPU est utilisé via les bibliothèques CUDA installées automatiquement par
uv — **pas besoin d'installer le CUDA Toolkit**. Il faut seulement un driver
NVIDIA raisonnablement récent (2023+). Vérifier :

```powershell
nvidia-smi
```

- Si un tableau s'affiche avec le nom du GPU : c'est bon.
- Si la commande est introuvable : installer le driver depuis
  <https://www.nvidia.com/fr-fr/drivers/> (ou via l'application « NVIDIA App »),
  puis redémarrer.

Rien d'autre à installer : ffmpeg n'est pas nécessaire (le décodage
audio/vidéo est embarqué dans les bibliothèques Python).

## 3. Copier le script

Créer un dossier et y copier `transcribe.py` (par clé USB, cloud, etc.) :

```powershell
mkdir $env:USERPROFILE\transcribe
# puis copier transcribe.py dans C:\Users\<votre-nom>\transcribe\
```

## 4. Installer le fichier d'identification Google

Récupérer le `credentials.json` (client OAuth « Desktop app » — voir la
section « Pré-requis Google Cloud » du [README](README.md) ; le même fichier
que sur le Mac fonctionne, il suffit de le copier).

Le placer dans `C:\Users\<votre-nom>\.transcribe\` — l'Explorateur Windows
refuse parfois de créer un dossier commençant par un point, donc via
PowerShell :

```powershell
mkdir $env:USERPROFILE\.transcribe
# puis copier credentials.json dans ce dossier, par exemple depuis Téléchargements :
copy $env:USERPROFILE\Downloads\credentials.json $env:USERPROFILE\.transcribe\
```

## 5. Vérifier l'installation

```powershell
cd $env:USERPROFILE\transcribe
uv run transcribe.py --check
```

Affiche un bilan de l'installation ; tout doit être ✓ (ou –) :

```
Vérification de l'environnement :
  ✓ Python 3.12.x
  ✓ faster-whisper x.y (ctranslate2 x.y)
  ✓ décodage audio/vidéo intégré (PyAV x.y, ffmpeg non requis)
  ✓ GPU CUDA détecté (1 périphérique(s))
  – modèle large-v3 absent du cache → téléchargé au premier lancement (~3 Go pour large-v3)
  ✓ client OAuth Google (C:\Users\...\.transcribe\credentials.json)
  – accès Drive pas encore autorisé → le navigateur s'ouvrira au lancement
```

Un ✗ indique ce qui manque (et où le corriger). Ce même bilan s'affiche au
début de chaque lancement.

## 6. Premier lancement

```powershell
uv run transcribe.py "https://drive.google.com/drive/folders/<ID-du-dossier>"
```

Au premier lancement, dans l'ordre (une seule fois chacun) :

1. uv télécharge Python (~1 min) puis les bibliothèques, y compris les DLL
   CUDA (~2 Go) ;
2. le navigateur s'ouvre pour autoriser l'accès à Google Drive (le jeton est
   ensuite mémorisé dans `.transcribe\token.json`) ;
3. le modèle Whisper large-v3 (~3 Go) est téléchargé et mis en cache.

Vérifier que cette ligne s'affiche — elle confirme que le GPU est utilisé :

```
Modèle large-v3 chargé sur cuda (float16)
```

Les lancements suivants démarrent directement.

## Dépannage

- **`Modèle ... chargé sur cpu (int8)`** : le GPU n'a pas été trouvé (driver
  absent/trop ancien ?). La transcription fonctionne quand même, mais
  beaucoup plus lentement. Vérifier `nvidia-smi` et mettre à jour le driver.
- **`chargé sur cuda (int8_float16)`** : normal si le GPU a moins de ~5 Go de
  VRAM — le script a basculé sur un mode plus économe, qualité quasi
  identique.
- **Le navigateur ne s'ouvre pas à l'étape OAuth** : copier l'URL affichée
  dans PowerShell et l'ouvrir manuellement.
- **Ré-autorisation demandée après 7 jours** : l'app OAuth Google est en mode
  « Testing » — la passer « In production » (voir README).
- **Purger le modèle du cache** (pour forcer un re-téléchargement propre,
  par exemple après un téléchargement corrompu) :

  ```powershell
  Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\hub\models--Systran--faster-whisper-large-v3"
  ```

  (sur macOS : `rm -rf ~/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3`)
