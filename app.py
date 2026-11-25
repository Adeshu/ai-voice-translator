import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import os
import torch
import time
import re
import matplotlib.pyplot as plt
import noisereduce as nr
from faster_whisper import WhisperModel
from TTS.api import TTS
from deep_translator import GoogleTranslator

# --- 1. PREMIUM UI CONFIG & CUSTOM CSS ---
st.set_page_config(page_title="Neural Voice Engine", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# Injecting heavy custom CSS to override Streamlit's default styling
st.markdown("""
    <style>
    /* Hide Streamlit default branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main App Background adjustment (works best if Streamlit is set to Dark Mode) */
    .stApp {
        background-color: #0E1117;
    }

    /* Premium Button Styling */
    div.stButton > button {
        background: linear-gradient(90deg, #3a7bd5 0%, #3a6073 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(58, 123, 213, 0.4);
        border: none;
        color: #ffffff;
    }

    /* Styling the Transcript Box */
    .transcript-box {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-left: 4px solid #3a7bd5;
        border-radius: 4px 8px 8px 4px;
        padding: 24px;
        margin-top: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .transcript-text {
        font-family: 'Inter', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #E2E8F0;
        line-height: 1.4;
    }

    /* Custom Gradient Text for Headers */
    .gradient-text {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. GPU LOADING ---
@st.cache_resource(show_spinner="Booting Neural Engine (Loading AI Models into VRAM)...")
def load_models():
    whisper = WhisperModel("small", device="cuda", compute_type="float16")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    return whisper, tts

whisper_model, tts_model = load_models()

# --- 3. HALLUCINATION FILTERS ---
BAD_PHRASES = ["thank you", "thanks for watching", "subtitles", "you", "bye"]

def is_hallucination(text):
    text = text.lower().strip()
    if len(text) < 2: return True
    if text in BAD_PHRASES: return True
    if text.startswith("(") and text.endswith(")"): return True
    if re.search(r'(.)\1{3,}', text): return True 
    vowels = sum(1 for char in text if char in 'aeiou')
    if len(text) > 0 and (vowels / len(text)) > 0.8: return True
    return False

# --- 4. MAIN APP HEADER ---
st.markdown('<p class="gradient-text">⚡ Neural Voice Engine</p>', unsafe_allow_html=True)
st.markdown("<p style='color: #94A3B8; font-size: 1.1rem;'>Real-time vocal cloning and translation powered by RTX 4050.</p>", unsafe_allow_html=True)
st.divider()

if "recording_active" not in st.session_state:
    st.session_state.recording_active = False

# --- 5. SIDEBAR SETTINGS ---
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")
    
    target_lang = st.selectbox(
        "🌐 Target Language", 
        options=["hi", "ja", "es", "fr", "de", "it", "pt"],
        format_func=lambda x: {"hi": "Hindi", "ja": "Japanese", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian", "pt": "Portuguese"}[x]
    )
    
    st.divider()
    st.markdown("### ⚙️ Audio Tuning")
    mic_threshold = st.slider("Mic Sensitivity", 0.01, 0.1, 0.05, 0.01, help="Increase if it picks up background noise.")
    
    st.divider()
    st.caption("Status: RTX Subsystems Online 🟢")

# --- 6. VOICE PROFILE SETUP ---
profile_expander = st.expander("👤 Step 1: Voice Profile Extraction (Required)", expanded=not os.path.exists("reference.wav"))
with profile_expander:
    st.markdown("<p style='color: #CBD5E1;'>Read this sentence loudly into your microphone: <b>'The quick brown fox jumps over the lazy dog.'</b></p>", unsafe_allow_html=True)
    if st.button("🎤 Extract Voice Signature"):
        fs = 16000
        with st.spinner("Recording... Speak now!"):
            rec = sd.rec(int(10 * fs), samplerate=fs, channels=1)
            sd.wait()
            
            trimmed_audio = rec.flatten()[int(0.25 * fs):]
            clean_ref = nr.reduce_noise(y=trimmed_audio, sr=fs, prop_decrease=0.4)
            
            max_amp = np.max(np.abs(clean_ref))
            if max_amp > 0:
                clean_ref = clean_ref / max_amp
            
            clean_ref = np.int16(clean_ref * 32767)
            write("reference.wav", fs, clean_ref)
        st.success("✨ Signature Extracted Successfully! You can close this panel.")

# --- 7. LIVE TRANSLATION DASHBOARD ---
st.markdown("### 🔴 Active Translation Stream")

col_controls, col_visuals = st.columns([1, 2.5])

with col_controls:
    start_btn = st.button("🚀 Initialize Stream", use_container_width=True)
    stop_btn = st.button("🛑 Terminate Stream", use_container_width=True)

    if start_btn: st.session_state.recording_active = True
    if stop_btn: 
        st.session_state.recording_active = False
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    status_container = st.container(border=True)
    status_box = status_container.empty()

with col_visuals:
    viz_container = st.empty()
    transcript_box = st.empty()
    
    # Placeholder state before recording starts
    if not st.session_state.recording_active:
        transcript_box.markdown(
            '<div class="transcript-box"><p class="transcript-text" style="color: #64748B;">Awaiting vocal input...</p></div>', 
            unsafe_allow_html=True
        )

# --- 8. ADAPTIVE LOOP ---
if st.session_state.recording_active:
    if not os.path.exists("reference.wav"):
        status_box.error("⚠️ Extract a Voice Signature first!")
        st.stop()

    while st.session_state.recording_active:
        fs = 16000
        chunk_size = 1024
        recording = []
        silent_chunks = 0
        is_speaking = False
        
        status_box.info(f"🎧 Listening... (Sensitivity: {mic_threshold})")
        
        with sd.InputStream(samplerate=fs, channels=1) as stream:
            while True:
                data, _ = stream.read(chunk_size)
                volume = np.abs(data).max()
                
                if volume > mic_threshold:
                    is_speaking = True
                    silent_chunks = 0
                elif is_speaking:
                    silent_chunks += 1
                
                if is_speaking:
                    recording.append(data)
                
                if is_speaking and silent_chunks > (1.2 * fs / chunk_size):
                    break
                if len(recording) * chunk_size > 15 * fs:
                    break
        
        if not recording: continue 
        audio_data = np.concatenate(recording).flatten()
        
        if len(audio_data) > fs * 0.5:
            cleaned_audio = nr.reduce_noise(y=audio_data, sr=fs, prop_decrease=0.4)
            
            max_amp = np.max(np.abs(cleaned_audio))
            if max_amp > 0:
                cleaned_audio = cleaned_audio / max_amp
            
            cleaned_audio = np.int16(cleaned_audio * 32767)
            write("input_temp.wav", fs, cleaned_audio)
            
            # Sleek Transparent Visualizer colored to match the new UI
            fig, ax = plt.subplots(figsize=(8, 1.2))
            ax.plot(cleaned_audio[::50] / 32767.0, color='#00C9FF', linewidth=1.5)
            ax.axis('off')
            fig.patch.set_alpha(0.0) 
            ax.patch.set_alpha(0.0)
            viz_container.pyplot(fig)
            plt.close(fig)

            status_box.warning("⚡ Processing Data...")
            
            segments, info = whisper_model.transcribe(
                "input_temp.wav", 
                beam_size=1,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                condition_on_previous_text=False
            )
            text = " ".join([s.text for s in segments]).strip()
            
            if is_hallucination(text):
                status_box.error("🔇 Filtered static.")
                continue 
            
            # Using the new CSS classes for the transcript output
            transcript_box.markdown(
                f'<div class="transcript-box"><p class="transcript-text">"{text}"</p></div>', 
                unsafe_allow_html=True
            )
            
            try:
                translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
                translated = re.sub(r'\d+', '', translated) 
                
                status_box.success("🗣️ Synthesizing...")
                tts_model.tts_to_file(
                    text=translated,
                    speaker_wav="reference.wav",
                    language=target_lang,
                    file_path="output.wav"
                )
                
                st.audio("output.wav", autoplay=True)
                time.sleep(1)
                
            except Exception as e:
                status_box.error(f"Error: {str(e)}")