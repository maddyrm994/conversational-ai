import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode, ClientSettings
from groq import Groq
import vosk
import json
import pydub
import io
import time
from gtts import gTTS
import base64
import queue
import av

# --- INITIALIZATION & SETUP ---

# Use a thread-safe queue to pass audio frames from the WebRTC thread to the main thread.
if "audio_frames_queue" not in st.session_state:
    st.session_state.audio_frames_queue = queue.Queue()

# VOSK Model Loading
@st.cache_resource
def load_vosk_model():
    model_path = "model/vosk-model-small-en-us-0.15"
    try:
        return vosk.Model(model_path)
    except Exception as e:
        st.error(f"Failed to load Vosk model. Error: {e}")
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
if "run_conversation" not in st.session_state:
    st.session_state.run_conversation = False

# --- AUDIO & AI FUNCTIONS ---

def text_to_audio_autoplay(text):
    """Converts text to an audio file and returns HTML for autoplaying it."""
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        b64 = base64.b64encode(mp3_fp.read()).decode()
        audio_html = f"""
            <audio autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            Your browser does not support the audio element.
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
    def recv(self, frame: av.AudioFrame):
        # Put the raw audio frame into our thread-safe queue
        st.session_state.audio_frames_queue.put(frame)
        return frame

# --- STREAMLIT UI ---

st.set_page_config(layout="wide", page_title="AI Voice Assistant (Cloud)")

# --- UI for Title and Clear Chat Button ---
col1, col2 = st.columns([5, 1])
with col1:
    st.title("️🎙️ AI Voice Assistant (Cloud Version)")
    st.markdown("This version runs on the cloud. Click 'START' in the box below and allow microphone access.")
with col2:
    if st.button("Clear Chat 🗑️", use_container_width=True):
        st.session_state.history = []
        # Clear the queue as well
        while not st.session_state.audio_frames_queue.empty():
            st.session_state.audio_frames_queue.get()
        st.rerun()

# Display conversation history
for user_msg, ai_msg in st.session_state.history:
    with st.chat_message("user"):
        st.write(user_msg)
    with st.chat_message("assistant"):
        st.write(ai_msg)

status_placeholder = st.empty()
audio_player_placeholder = st.empty()

# The WebRTC component that accesses the microphone
webrtc_ctx = webrtc_streamer(
    key="audio-recorder",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    client_settings=ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True},
    ),
    # The 'send_interval' argument has been removed from this call
)

if webrtc_ctx.state.playing and not st.session_state.run_conversation:
    status_placeholder.info("🎙️ Microphone is active. Speak your message, then press 'Stop and Process'.")
    if st.button("🛑 Stop and Process", type="primary", use_container_width=True):
        st.session_state.run_conversation = True
        st.rerun()

elif not webrtc_ctx.state.playing:
    status_placeholder.warning("🎤 Microphone is off. Please click 'START' in the component box above to activate it.")

# This block runs only AFTER "Stop and Process" is clicked
if st.session_state.run_conversation:
    status_placeholder.info("Processing audio...")
    
    audio_frames = []
    while not st.session_state.audio_frames_queue.empty():
        audio_frames.append(st.session_state.audio_frames_queue.get())
    
    if audio_frames:
        full_audio_segment = pydub.AudioSegment.empty()
        for frame in audio_frames:
            sound = pydub.AudioSegment(
                data=frame.to_ndarray().tobytes(),
                sample_width=frame.format.bytes,
                frame_rate=frame.sample_rate,
                channels=len(frame.layout.channels),
            )
            full_audio_segment += sound

        if len(full_audio_segment) > 0:
            full_audio_segment = full_audio_segment.set_channels(1).set_frame_rate(16000)
            audio_data = full_audio_segment.raw_data

            with st.spinner("Transcribing your speech..."):
                recognizer = vosk.KaldiRecognizer(vosk_model, 16000)
                recognizer.AcceptWaveform(audio_data)
                result_json = recognizer.FinalResult()
                result_dict = json.loads(result_json)
                transcribed_text = result_dict.get("text", "").strip()

            if transcribed_text:
                st.success(f"Transcription: '{transcribed_text}'")
                with st.spinner("Generating AI response..."):
                    ai_response = generate_response(transcribed_text)
                
                st.session_state.history.append((transcribed_text, ai_response))
                audio_html = text_to_audio_autoplay(ai_response)
                if audio_html:
                    audio_player_placeholder.markdown(audio_html, unsafe_allow_html=True)
                
                st.session_state.run_conversation = False
                time.sleep(1) 
                st.rerun()
            else:
                st.warning("No speech was detected. Please try again.")
                st.session_state.run_conversation = False
                time.sleep(3); st.rerun()
        else:
            st.warning("No audio was captured. Please try again.")
            st.session_state.run_conversation = False
            time.sleep(3); st.rerun()
    else:
        st.warning("Audio buffer is empty. Please try recording again.")
        st.session_state.run_conversation = False
        time.sleep(3); st.rerun()

# Sidebar
st.sidebar.header("About")
st.sidebar.info("This is the cloud-ready version of the AI Voice Assistant, using thread-safe data handling for stability.")
