import streamlit as st
import speech_recognition as sr
import pyttsx3
import time
from groq import Groq
import json
from vosk import Model, KaldiRecognizer

# --- INITIALIZATION ---

# Initialize TTS engine
try:
    engine = pyttsx3.init()
except (ImportError, RuntimeError):
    st.error("Text-to-Speech (pyttsx3) engine failed to initialize.")
    st.stop()

# Initialize Speech Recognizer (we still use it for microphone access)
r = sr.Recognizer()

# Initialize Groq API Client
try:
    api_key = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=api_key)
except KeyError:
    st.error("Groq API key not found. Please add it to your Streamlit secrets.")
    st.stop()

# --- VOSK MODEL LOADING ---
# Use Streamlit's caching to load the model only once.
@st.cache_resource
def load_vosk_model():
    model_path = "model/vosk-model-small-en-us-0.15"
    try:
        return Model(model_path)
    except Exception as e:
        st.error(f"Failed to load model from {model_path}. Please ensure the model is downloaded and in the correct path.")
        st.error(f"Error: {e}")
        return None

vosk_model = load_vosk_model()
if vosk_model is None:
    st.stop()

# --- CORE FUNCTIONS ---

def speak_text(text):
    """Converts text to speech."""
    try:
        st.write(f"{text}")
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.error(f"Error in text-to-speech: {e}")

def listen_and_transcribe(status_placeholder):
    """Listens for audio via microphone and transcribes it using Vosk."""
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.5)
        status_placeholder.info("🎙️ Listening...")
        
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
            status_placeholder.info("🧠 Processing with Transcriber...")
            
            # Get raw audio data from SpeechRecognition's AudioData object
            raw_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
            
            # Create a Vosk recognizer
            recognizer = KaldiRecognizer(vosk_model, 16000)
            
            if recognizer.AcceptWaveform(raw_data):
                result_json = recognizer.Result()
                # The result is a JSON string, parse it to get the text
                result_dict = json.loads(result_json)
                text = result_dict.get("text", "")
                
                if text:
                    status_placeholder.success("✅ Transcription Complete!")
                    time.sleep(1)
                    return text.lower()
                else:
                    status_placeholder.warning("🤔 No speech detected in the audio.")
                    return None
            else:
                # This part handles partial results if needed, but for this app, we take the final
                partial_result_json = recognizer.PartialResult()
                partial_dict = json.loads(partial_result_json)
                partial_text = partial_dict.get("partial", "")
                if partial_text:
                    status_placeholder.success("✅ Transcription Complete! ")
                    time.sleep(1)
                    return partial_text.lower()
                else:
                    status_placeholder.warning("🤔 Could not get a final result from Transcriber.")
                    return None

        except sr.WaitTimeoutError:
            status_placeholder.warning("👂 Listening timed out.")
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred during transcription: {e}")
            return None

def generate_response(user_text):
    """Generates a response using the Groq API."""
    st.info("💡 Getting response from AI...")
    system_prompt = "You are a helpful and friendly voice assistant. Your name is AI. Keep your responses concise, conversational, and suitable for being spoken aloud."
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            model="llama3-8b-8192",
            temperature=0.7, max_tokens=250,
        )
        response = chat_completion.choices[0].message.content
        st.success("✅ Response received!")
        time.sleep(1)
        return response
    except Exception as e:
        st.error(f"An error occurred with the API: {e}")
        return "I'm sorry, I'm having trouble connecting to my brain right now."

# --- STREAMLIT UI (No changes needed below this line) ---

st.set_page_config(layout="wide", page_title="Conversational AI")
st.title("️🎙️ Conversational AI")
st.markdown("Click the button and speak. The AI will listen, transcribe, and respond.")

if 'history' not in st.session_state:
    st.session_state.history = []

for user_msg, ai_msg in st.session_state.history:
    with st.chat_message("user"):
        st.write(user_msg)
    with st.chat_message("assistant"):
        st.write(ai_msg)

status_placeholder = st.empty()

if st.button("Start Listening", type="primary", use_container_width=True):
    user_input = listen_and_transcribe(status_placeholder)
    
    if user_input:
        status_placeholder.empty()
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            ai_response = generate_response(user_input)
            speak_text(ai_response)
        st.session_state.history.append((user_input, ai_response))
        st.rerun()
