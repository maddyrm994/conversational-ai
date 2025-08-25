import streamlit as st
import speech_recognition as sr
import pyttsx3
import time
from groq import Groq
import json
from vosk import Model, KaldiRecognizer
import multiprocessing

# --- TTS Subprocess Target Function ---
# This function will run in a separate process.
# It MUST be a top-level function and cannot use Streamlit objects like st.error.
def tts_subprocess_target(text_to_speak):
    """
    Initializes a TTS engine in a new process, speaks the text, and terminates.
    """
    try:
        engine = pyttsx3.init()
        engine.say(text_to_speak)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        # If something goes wrong in the subprocess, print to the console.
        print(f"Error in TTS subprocess: {e}")

# --- Main App Functions ---

# We now have a new speak_text function that launches the subprocess
def speak_text(text):
    """
    Launches a separate process to handle text-to-speech.
    """
    p = multiprocessing.Process(target=tts_subprocess_target, args=(text,))
    p.start()
    # We don't call p.join(), allowing the Streamlit app to continue running.

@st.cache_resource
def load_vosk_model():
    model_path = "model/vosk-model-small-en-us-0.15"
    try:
        return Model(model_path)
    except Exception:
        st.error("Failed to load Speech-To-Text model. Ensure it's in the 'model' directory.")
        return None

def listen_and_transcribe(r, status_placeholder, vosk_model):
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.5)
        status_placeholder.info("🎙️ Listening... Speak now!")
        try:
            audio = r.listen(source, timeout=10, phrase_time_limit=15)
            status_placeholder.info("🧠 Processing with Speech-To-Text model...")
            raw_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
            recognizer = KaldiRecognizer(vosk_model, 16000)
            if recognizer.AcceptWaveform(raw_data):
                result_json = recognizer.Result()
                result_dict = json.loads(result_json)
                text = result_dict.get("text", "")
                if text:
                    status_placeholder.success("✅ Transcription Complete!")
                    time.sleep(1)
                    return text.lower()
            partial_json = recognizer.PartialResult()
            partial_dict = json.loads(partial_json)
            partial_text = partial_dict.get("partial", "")
            if partial_text:
                status_placeholder.success("✅ Transcription Complete!")
                time.sleep(1)
                return partial_text.lower()
            status_placeholder.warning("🤔 No clear speech detected.")
            return None
        except sr.WaitTimeoutError:
            status_placeholder.warning("👂 Listening timed out. No speech was detected.")
            return None
        except Exception as e:
            st.error(f"An error occurred during transcription: {e}")
            return None

def generate_response(client, user_text):
    st.info("💡 Getting response from AI...")
    system_prompt = "You are a helpful and friendly voice assistant. Your name is AI. Keep your responses concise, conversational, and suitable for being spoken aloud."
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}],
            model="llama3-8b-8192", temperature=0.7, max_tokens=250)
        response = chat_completion.choices[0].message.content
        st.success("✅ AI response received!")
        time.sleep(1)
        return response
    except Exception as e:
        st.error(f"An error occurred with the AI API: {e}")
        return "I'm sorry, I'm having trouble connecting to my brain right now."


# --- Main Application Logic ---
# It's good practice to wrap the main app logic in a function and use a __name__ == "__main__" block
def main():
    st.set_page_config(layout="wide", page_title="Conversational AI", page_icon="🎙️")

    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Load resources
    vosk_model = load_vosk_model()
    if vosk_model is None:
        st.stop()
    
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        client = Groq(api_key=api_key)
    except KeyError:
        st.error("AI API key not found. Please add it to your Streamlit secrets.")
        st.stop()

    r = sr.Recognizer()

    # UI for Title and Clear Chat Button
    col1, col2 = st.columns([5, 1])
    with col1:
        st.title("️🎙️ Conversational AI")
        st.markdown("Your offline-first voice companion.")
    with col2:
        if st.button("Clear Chat 🗑️", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    # Display conversation history
    for user_msg, ai_msg in st.session_state.history:
        with st.chat_message("user"):
            st.write(user_msg)
        with st.chat_message("assistant"):
            st.write(ai_msg)

    # Placeholder for status messages
    status_placeholder = st.empty()

    # Main Interaction Logic
    if st.button("🎤 Start Speaking", type="primary", use_container_width=True):
        user_input = listen_and_transcribe(r, status_placeholder, vosk_model)
        
        if user_input:
            status_placeholder.empty()
            ai_response = generate_response(client, user_input)
            status_placeholder.empty()

            # Update history and UI, then speak
            st.session_state.history.append((user_input, ai_response))
            
            # Launch the TTS in a separate process
            speak_text(ai_response)
            
            # Rerun to display the new messages immediately
            st.rerun()
        else:
            status_placeholder.empty()

# This block is crucial for multiprocessing to work reliably on all platforms
if __name__ == "__main__":
    # On Windows, the 'spawn' start method is the default. This is needed.
    multiprocessing.freeze_support() 
    main()
