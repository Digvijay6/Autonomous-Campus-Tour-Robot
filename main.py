import os
import struct
import json
import time
import platform
import pvporcupine
import pyaudio
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from openai import AzureOpenAI
from campus_info import get_campus_info
from irrelevant import handle_irrelevant
# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

# Azure Configuration
SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
AZURE_OPEN_AI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPEN_AI_ENDPOINT = os.getenv("ENDPOINT_URL")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

# Picovoice
PV_ACCESS_KEY = os.getenv("PV_ACCESS_KEY")


# -----------------------------
# Azure SDK Clients
# -----------------------------
client = AzureOpenAI(
    api_key=AZURE_OPEN_AI_KEY,
    api_version="2024-05-01-preview",
    azure_endpoint=AZURE_OPEN_AI_ENDPOINT
)


# -----------------------------
# SDK-based Helper Functions
# -----------------------------
def speak(text):
    """Convert text to speech using Azure SDK"""
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
    result = synthesizer.speak_text_async(text).get()
    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("TTS failed:", result.reason)


def recognize_from_mic():
    """Recognize speech from mic using Azure SDK"""
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

    print("Listening...")
    result = recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized:", result.text)
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized.")
    elif result.reason == speechsdk.ResultReason.Canceled:
        print("Recognition canceled:", result.cancellation_details.reason)
    return None


def call_azure_openai_sdk(command_text):
    """Call Azure OpenAI Chat Completion using new SDK"""
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

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": command_text}
            ],
            temperature=0.2,
        )
        reply = response.choices[0].message.content
        print("OpenAI response:", reply)
        return json.loads(reply)
    except Exception as e:
        print("Azure OpenAI SDK Error:", e)
        return None


def process_command(parsed_command):
    """Handle parsed commands"""
    if not parsed_command or "action" not in parsed_command:
        return "Sorry, I didn't understand that."

    action = parsed_command["action"]
    params = parsed_command.get("parameters", {})

    if action == "navigate":
        return f"Navigating to {params.get('destination', 'your destination')}."
    elif action == "campus_info":
        return get_campus_info(params.get("query", ""))
    elif action == "image_analysis":
        return "Analyzing current location..."
    elif action == "irrelevant":
        return handle_irrelevant(params.get("query", ""))
    else:
        return "Unknown command."

def get_wakeword_path():
    system = platform.system().lower()

    # For Raspberry Pi (Linux ARM)
    if system == "linux":
        return "hellum_pi.ppn"

    # For Windows development
    return "hellum_win.ppn"


# -----------------------------
# Main Wake Word Loop
# -----------------------------
def main():
    # Initialize wake word engine
    try:
        porcupine = pvporcupine.create(
            access_key=PV_ACCESS_KEY,
            keyword_paths=[get_wakeword_path()],
            sensitivities=[0.7]
        )
    except Exception as e:
        print("Wake word init error:", e)
        return

    # Audio stream setup
    pa = pyaudio.PyAudio()
    print("Mic ready:", pa.get_default_input_device_info())
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
            result = porcupine.process(pcm_unpacked)

            if result >= 0:
                print("\nWake word detected!")
                speak("Hi, I am Hellum, your campus tour bot.")

                command = recognize_from_mic()
                if command:
                    parsed = call_azure_openai_sdk(command)
                    response = process_command(parsed)
                    print("Response:", response)
                    speak(response)

                print("Ready for the next command...")

                time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down...")

    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        porcupine.delete()


if __name__ == "__main__":
    main()
