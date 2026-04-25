# 🎙 Darija → Français — Traducteur temps réel

Traduction vocale **Darija marocain → Français** pour conférences.
Pipeline 100% open-source et gratuit : **Whisper + Aya (Ollama) + Coqui TTS**.

---

## 🚀 Installation (10 minutes)

### 1. Prérequis
- Python 3.10+
- [Ollama](https://ollama.ai) installé
- ffmpeg : `sudo apt install ffmpeg` ou `brew install ffmpeg`

### 2. Installer les dépendances Python
```bash
pip install -r requirements.txt
```

### 3. Télécharger le modèle LLM (traduction)
```bash
ollama pull aya:8b
```
> Alternative plus légère : `ollama pull aya:4b`

### 4. Lancer le serveur
```bash
python server.py
```
Le serveur démarre sur **http://0.0.0.0:8000**

---

## 📱 Utiliser sur téléphone

1. PC et téléphone sur **le même WiFi**
2. Trouve l'IP locale de ton PC :
   - Windows : `ipconfig` → IPv4
   - Mac/Linux : `ifconfig` ou `ip addr`
   - Exemple : `192.168.1.42`
3. Ouvre `index.html` sur le téléphone (ou sers-le avec un petit serveur HTTP)
4. Dans **Réglages**, entre : `http://192.168.1.42:8000`
5. C'est prêt ! 🎉

### Servir le HTML depuis le PC :
```bash
# Dans le dossier du projet
python -m http.server 3000
# Puis ouvre http://192.168.1.42:3000 sur le téléphone
```

---

## ⚙️ Routes API

| Route | Méthode | Description |
|-------|---------|-------------|
| `GET /` | GET | Santé du serveur |
| `POST /stt` | POST | Audio → Texte Darija (Whisper) |
| `POST /translate` | POST | Darija → Français (Aya LLM) |
| `POST /tts` | POST | Français → Audio WAV (Coqui) |
| `POST /pipeline` | POST | Pipeline complet en une requête |

---

## 🎛 Config modèles (server.py)

```python
WHISPER_MODEL = "large-v3"   # medium si GPU < 6Go
OLLAMA_MODEL  = "aya:8b"     # aya:4b si RAM < 8Go
TTS_MODEL     = "tts_models/fr/mai/tacotron2-DDC"
```

---

## 💻 Config matérielle recommandée

| Usage | GPU VRAM | RAM |
|-------|----------|-----|
| Confortable (large-v3) | 8 Go+ | 16 Go+ |
| Correct (medium) | 4 Go+ | 8 Go+ |
| CPU uniquement (base) | — | 8 Go+ |
