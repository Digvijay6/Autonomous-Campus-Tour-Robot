import os
import struct
import json
import time
import requests
import pvporcupine
import pyaudio
from dotenv import load_dotenv
import tempfile
import winsound
import platform


from campus_info import get_campus_info
from irrelevant import handle_irrelevant

# -----------------------------
# Configuration and Constants
# -----------------------------
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPEN_AI_ENDPOINT = os.getenv("ENDPOINT_URL")
AZURE_OPEN_AI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

# Azure Speech Configuration
SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")

# Picovoice Configuration
PV_ACCESS_KEY = os.getenv("PV_ACCESS_KEY")

# Audio Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_SIZE = 1024
RECORD_SECONDS = 8


# -----------------------------
# Helper Functions
# -----------------------------
def get_wakeword_path():
    system = platform.system().lower()

    # For Raspberry Pi (Linux ARM)
    if system == "linux":
        return "hellum_pi.ppn"

    # For Windows development
    return "hellum_win.ppn"

def get_speech_token():
    """Get Azure Speech auth token"""
    url = f"https://{SPEECH_REGION}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
    headers = {"Ocp-Apim-Subscription-Key": SPEECH_KEY}

    try:
        response = requests.post(url, headers=headers)
        return response.text if response.status_code == 200 else None
    except Exception as e:
        print(f"Token Error: {e}")
        return None


def speak(text):
    """Convert text to speech using Azure REST API"""
    token = get_speech_token()
    if not token:
        return

    # SSML Configuration
    ssml = f"""
    <speak version='1.0' xml:lang='en-US'>
        <voice name='en-US-JennyNeural'>
            {text}
        </voice>
    </speak>
    """

    # API Request
    url = f"https://{SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "riff-16khz-16bit-mono-pcm"  # WAV format
    }

    try:
        response = requests.post(url, headers=headers, data=ssml)
        if response.status_code == 200:
            # Save to temp file and play
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(response.content)
                f.flush()
                winsound.PlaySound(f.name, winsound.SND_FILENAME)
        else:
            print(f"TTS Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Speech Error: {e}")

def call_azure_open_ai(command_text):
    """Call Azure OpenAI using REST API"""
    print("sending command to open ai")
    url = f"{AZURE_OPEN_AI_ENDPOINT}/openai/deployments/{DEPLOYMENT_NAME}/chat/completions?api-version=2024-05-01-preview"

    system_prompt = """
        You are a command interpreter for a campus tour autonomous car.
        Given the user's command, parse it and return a JSON object with the following keys:
        - "action": one of "navigate", "campus_info", "image_analysis", or "irrelevant".
        - "query": the input command.
        - "parameters": a JSON object containing additional details if applicable.
        Examples:
        - For "take me to the library", return {"action": "navigate", "parameters": {"destination": "library", "query": "take me to the library"}}.
        - For "tell me about this university", return {"action": "campus_info", "parameters": {"query": "Tell me about the hostel facilities"}}.
        - For "where are we", return {"action": "image_analysis", "parameters": {"query": "where are we right now?"}}.
        - For irrelevant commands like "take me to Paris", return {"action": "irrelevant" "parameters": {"query": "take me to paris"}}.
        Now parse the following command: {command}
"""

    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPEN_AI_KEY
    }

    data = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": command_text}
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        llm_reply = response.json()["choices"][0]["message"]["content"]
        print(f"Open AI response : {llm_reply}")
        return json.loads(llm_reply)
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return None


def azure_speech_to_text(audio_data):
    """Convert audio to text using Azure Speech REST API"""
    print("Sending audio to STT model")
    language = "en-US"
    url = f"https://{SPEECH_REGION}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language={language}"

    headers = {
        "Ocp-Apim-Subscription-Key": SPEECH_KEY,
        "Content-Type": "audio/wav; codec=audio/pcm; samplerate=16000",
        "Accept": "application/json"
    }

    try:
        response = requests.post(url, headers=headers, data=audio_data)
        print(f"STT response: {response.json()}")
        if response.status_code == 200:
            return response.json()["DisplayText"]
        return None
    except Exception as e:
        print(f"Speech API Error: {e}")
        return None


def record_audio():
    """Record audio from microphone to memory buffer"""
    print('Recording audio')
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )

    frames = []
    for _ in range(0, int(RATE / CHUNK_SIZE * RECORD_SECONDS)):
        frames.append(stream.read(CHUNK_SIZE))

    stream.stop_stream()
    stream.close()
    audio.terminate()
    print('Audio recorded')

    return b''.join(frames)


# -----------------------------
# Command Processing
# -----------------------------

def process_command(parsed_command):
    """Handle parsed commands"""
    print("Sending command to take action")
    print(f"Command : {parsed_command}")
    if not parsed_command or "action" not in parsed_command:
        return "Sorry, I didn't understand that"

    action = parsed_command["action"]
    params = parsed_command.get("parameters", {})

    if action == "navigate":
        return f"Navigating to {params.get('destination', 'your destination')}"
    elif action == "campus_info":
        return get_campus_info(params.get("query", ""))
    elif action == "image_analysis":
        return "Analyzing current location..."
    elif action == "irrelevant":
        return handle_irrelevant(params.get("query", ""))
    else:
        return "Unknown command"


# -----------------------------
# Main Wake Word Loop
# -----------------------------

def main():
    # Initialize wake word engine
    try:
        porcupine = pvporcupine.create(
            access_key=PV_ACCESS_KEY,
            keyword_paths=[get_wakeword_path()]
        )
    except Exception as e:
        print(f"Wake word init error: {e}")
        return

    # Audio stream setup
    pa = pyaudio.PyAudio()
    stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )

    print("System ready. Say 'Hellum' to activate...")

    try:
        while True:
            pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm_unpacked = struct.unpack_from("h" * porcupine.frame_length, pcm)

            if porcupine.process(pcm_unpacked) >= 0:
                print("\nWake word detected!")
                speak("Hi, I am Hellum, your campus tour bot")
                audio_data = record_audio()
                command = azure_speech_to_text(audio_data)

                if command:
                    parsed = call_azure_open_ai(command)
                    if parsed:
                        response = process_command(parsed)
                        print(response)
                        speak(response)
                        print("Completed command, Say 'Hellum' and ask next question")

                time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stream.close()
        pa.terminate()
        porcupine.delete()


if __name__ == "__main__":
    main()