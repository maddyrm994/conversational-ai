import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
from groq import Groq
import vosk
import json
import pydub
import io
import time
from gtts import gTTS
import base64

# --- INITIALIZATION ---

# VOSK Model Loading
@st.cache_resource
def load_vosk_model():
    model_path = "model/vosk-model-small-en-us-0.15"
    try:
        return vosk.Model(model_path)
    except Exception as e:
        st.error(f"Failed to load Vosk model from {model_path}. Ensure it's in your repo. Error: {e}")
        return None

vosk_model = load_vosk_model()

# Groq API Client
try:
    api_key = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=api_key)
except KeyError:
    st.error("Groq API key not found. Please add it to your Streamlit secrets.")
    st.stop()

# --- SESSION STATE ---
if "history" not in st.session_state:
    st.session_state.history = []
if "audio_buffer" not in st.session_state:
    st.session_state.audio_buffer = b""
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""
if "processing" not in st.session_state:
    st.session_state.processing = False

# --- AUDIO & AI FUNCTIONS ---

def text_to_audio_autoplay(text):
    """Converts text to an audio file and returns HTML for autoplaying it."""
    try:
        tts = gTTS(text=text, lang='en')
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        # Encode to base64
        b64 = base64.b64encode(mp3_fp.read()).decode()
        # Create the HTML audio player
        audio_html = f"""
            <audio autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        return audio_html
    except Exception as e:
        st.error(f"Error in TTS generation: {e}")
        return None

def generate_response(user_text):
    system_prompt = "You are a helpful and friendly voice assistant. Keep your responses concise and conversational."
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            model="llama3-8b-8192",
            temperature=0.7, max_tokens=250,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error with Groq API: {e}")
        return "Sorry, I'm having trouble connecting to my brain right now."

# --- WEBRTC AUDIO PROCESSOR ---

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_buffer = b""
        self.vosk_recognizer = vosk.KaldiRecognizer(vosk_model, 16000)

    def recv(self, frame):
        # The frame comes from the browser. Convert it to the right format for Vosk.
        # This assumes the browser sends audio at 48kHz. We'll downsample to 16kHz.
        try:
            audio = pydub.AudioSegment.from_raw(
                io.BytesIO(frame.to_ndarray().tobytes()),
                sample_width=frame.format.bytes,
                frame_rate=frame.sample_rate,
                channels=len(frame.layout.channels),
            )
            audio = audio.set_channels(1).set_frame_rate(16000)
            
            # Append to buffer if recording is active
            if st.session_state.get("is_recording", False):
                st.session_state.audio_buffer += audio.raw_data

        except Exception as e:
            st.error(f"Error processing audio frame: {e}")
        
        return frame

# --- STREAMLIT UI ---

st.set_page_config(layout="wide", page_title="AI Voice Assistant (Cloud)")

# --- UI for Title and Clear Chat Button ---
col1, col2 = st.columns([5, 1])
with col1:
    st.title("️🎙️ AI Voice Assistant (Cloud Version)")
    st.markdown("This version runs on the cloud. Click 'Start Recording' and allow microphone access.")
with col2:
    if st.button("Clear Chat 🗑️", use_container_width=True):
        st.session_state.history = []
        st.session_state.transcribed_text = ""
        st.rerun()

# Display conversation history
for user_msg, ai_msg in st.session_state.history:
    with st.chat_message("user"):
        st.write(user_msg)
    with st.chat_message("assistant"):
        st.write(ai_msg)

# Placeholder for status and audio player
status_placeholder = st.empty()
audio_player_placeholder = st.empty()

# The WebRTC component that accesses the microphone
# It's always "running" in the background, but we control data collection with a session state flag.
webrtc_ctx = webrtc_streamer(
    key="audio-recorder",
    mode=WebRtcMode.SEND_ONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"video": False, "audio": True},
)

# --- Main Interaction Logic ---
# Control buttons
c1, c2 = st.columns(2)
with c1:
    if not st.session_state.get("is_recording", False):
        if st.button("🎤 Start Recording", type="primary", use_container_width=True):
            st.session_state.is_recording = True
            st.session_state.audio_buffer = b"" # Reset buffer
            st.rerun()
    else:
        if st.button("🛑 Stop Recording", type="secondary", use_container_width=True):
            st.session_state.is_recording = False
            st.session_state.processing = True
            st.rerun()

if st.session_state.get("is_recording", False):
    status_placeholder.info("🎙️ Recording... Click 'Stop Recording' when you're done.")

# This block runs AFTER "Stop Recording" is clicked
if st.session_state.get("processing", False):
    with status_placeholder.container():
        with st.spinner("Transcribing your speech..."):
            if st.session_state.audio_buffer and vosk_model:
                recognizer = vosk.KaldiRecognizer(vosk_model, 16000)
                recognizer.AcceptWaveform(st.session_state.audio_buffer)
                result_json = recognizer.FinalResult()
                result_dict = json.loads(result_json)
                st.session_state.transcribed_text = result_dict.get("text", "").strip()
            else:
                st.session_state.transcribed_text = ""

        if st.session_state.transcribed_text:
            st.success(f"Transcription: '{st.session_state.transcribed_text}'")
            with st.spinner("Generating AI response..."):
                ai_response = generate_response(st.session_state.transcribed_text)
            
            # Add to history and generate audio
            st.session_state.history.append((st.session_state.transcribed_text, ai_response))
            audio_html = text_to_audio_autoplay(ai_response)
            if audio_html:
                audio_player_placeholder.markdown(audio_html, unsafe_allow_html=True)
            
            # Reset states for the next turn
            st.session_state.transcribed_text = ""
            st.session_state.audio_buffer = b""
            st.session_state.processing = False
            time.sleep(1) # Give a moment for audio to start playing
            st.rerun()
        else:
            st.warning("No speech was detected. Please try recording again.")
            st.session_state.processing = False
            time.sleep(2)
            st.rerun()

# Sidebar Information
st.sidebar.header("About")
st.sidebar.info(
    "This app uses a web-friendly architecture:\n"
    "1. **Mic Access:** `streamlit-webrtc` (in browser)\n"
    "2. **Speech-to-Text:** `Vosk` (on server)\n"
    "3. **LLM Inference:** `Groq API`\n"
    "4. **Text-to-Speech:** `gTTS` (generates audio on server)\n"
    "5. **Audio Playback:** HTML5 Audio (in browser)"
)
