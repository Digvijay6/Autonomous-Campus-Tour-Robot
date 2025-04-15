import pvporcupine
import pyaudio
import struct
import os
from dotenv import load_dotenv

load_dotenv()

porcupine = pvporcupine.create(
    access_key=os.getenv("PV_ACCESS_KEY"),
    keyword_paths=["hellum_win.ppn"],  # or your .ppn path
    sensitivities=[0.7]
)

pa = pyaudio.PyAudio()
print("Mic ready:", pa.get_default_input_device_info())

stream = pa.open(
    rate=porcupine.sample_rate,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=porcupine.frame_length
)

print("Say the wakeword...")

try:
    while True:
        pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

        if porcupine.process(pcm) >= 0:
            print("Wakeword detected!")
except KeyboardInterrupt:
    pass
finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
    porcupine.delete()
