import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from flask import Flask, request, send_file
import json
import io

# Load config
with open("config.json", "r") as file:
    config = json.load(file)

# Configs
device = config.get("device")
if not device:
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

min_p = config.get("minP", 0.05)
top_p = config.get("topP", 1)
repetition_penalty = config.get("repetitionPenalty", 1.2)
cfg_weight = config.get("cfgWeight", 0.5)
exaggeration = config.get("exaggeration", 0.5)
temperature = config.get("temperature", 0.8)
audio_prompt = config.get("audioPrompt", "input.wav")
audio_format = config.get("audioFormat", "wav")
audio_mime_type = config.get("audioMimeType", "audio/wav")
host = config.get("host")
port = config.get("port")

# Load model
print(f"Loading model with {device.upper()}...")
model = ChatterboxTTS.from_pretrained(device=device)

# Create flask app
app = Flask(__name__)

# /v1/text-to-speech/:voice_id
@app.route("/v1/text-to-speech/<voice_id>", methods=["POST"])
def text_to_speech(voice_id):
    data = request.get_json()
    text = data.get("text")
    audio_buffer = io.BytesIO()

    print(f"Generating speech for text \"{text}\"...")

    generated = model.generate(
        text=text,
        audio_prompt_path=audio_prompt,
        min_p=min_p,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        cfg_weight=cfg_weight,
        exaggeration=exaggeration,
        temperature=temperature
    )

    ta.save(
        uri=audio_buffer,
        src=generated,
        sample_rate=model.sr,
        format=audio_format,
    )
    audio_buffer.seek(0)
    
    return send_file(audio_buffer, mimetype=audio_mime_type)

if __name__ == "__main__":
    # Start server
    app.run(host=host, port=port)