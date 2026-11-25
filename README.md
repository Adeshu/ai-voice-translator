# AI Voice Translator (Neural Voice Engine)

Real-time speech-to-speech translation desktop app. It records your voice, transcribes it with Whisper, translates the text, and speaks the translation back in your own cloned voice.

## Features

- Live microphone streaming with simple voice-activity detection.
- Background-noise reduction before transcription.
- Hallucination filtering for short or repeated Whisper output.
- Voice-profile extraction so the translated speech keeps your timbre.
- Seven target languages: Hindi, Japanese, Spanish, French, German, Italian, Portuguese.
- Waveform visualiser and auto-playing translated audio.
- GPU accelerated (CUDA + float16) by default.

## Tech stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Speech recognition | faster-whisper (`small`, CUDA, float16) |
| Translation | deep-translator (Google) |
| Speech synthesis | Coqui TTS `xtts_v2` (voice cloning) |
| Audio I/O | sounddevice, SciPy |
| Noise reduction | noisereduce |
| Runtime | Python 3.10, PyTorch |

## Project structure

```text
ai-voice-translator/
├── app.py              # Streamlit app and streaming loop
├── utils.py            # Helper functions
├── requirements.txt
├── reference.wav       # Generated voice profile (ignored)
├── input_temp.wav      # Temporary recording (ignored)
├── output.wav          # Synthesised translation (ignored)
└── venv_310/           # Local virtual environment (ignored)
```

## Prerequisites

- Python 3.10
- A working microphone
- NVIDIA GPU with CUDA for the default configuration
- FFmpeg on `PATH` (recommended for audio formats)

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The first run downloads the Whisper and XTTS model weights, so it needs internet access and a few GB of disk space.

## Run

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open <http://localhost:8501>.

## Usage

1. Expand **Step 1: Voice Profile Extraction** and read the displayed sentence into the microphone.
2. Choose a target language in the sidebar.
3. Adjust microphone sensitivity if the app picks up background noise.
4. Click **Initialize Stream** and speak. The transcript, translation and cloned-voice audio appear in the dashboard.
5. Click **Terminate Stream** to stop.

## Configuration

| Setting | Where | Default |
|---|---|---|
| Target language | Sidebar | Hindi (`hi`) |
| Microphone threshold | Sidebar | `0.05` |
| Whisper model | `app.py` (`load_models`) | `small` |
| Device | `app.py` | `cuda` |
| TTS model | `app.py` | `xtts_v2` |

If you do not have a CUDA GPU, change `device="cuda"` to `device="cpu"` and `compute_type="float16"` to `compute_type="int8"` in `load_models()`.

## Performance notes

- CPU-only inference is noticeably slower; a GPU is strongly recommended.
- Streaming latency depends on utterance length and model load time.
- The first request is slow because models are loaded into memory/VRAM.

## Limitations

- Translation uses an unofficial Google endpoint and needs internet access.
- Voice cloning should only be used with the speaker's consent.
- No `.env` is required; no API keys are stored in this project.
- Temporary `.wav` files and the 5.8 GB virtual environment are git-ignored.

## Licence

Educational project. Third-party models keep their own licences.
