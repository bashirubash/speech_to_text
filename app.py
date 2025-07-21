from flask import Flask, request, jsonify, render_template_string
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import librosa
import numpy as np

app = Flask(__name__)

# Load model from Hugging Face Hub
model_name = "facebook/wav2vec2-large-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)
device = torch.device("cpu")
model.to(device)

# HTML template for simple UI
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Speech to Text Converter</title>
</head>
<body style="text-align: center; padding-top: 50px;">
    <h1>Speech to Text Converter</h1>
    <form method="POST" action="/transcribe" enctype="multipart/form-data">
        <input type="file" name="audio_file" accept=".wav,.mp3" required><br><br>
        <input type="submit" value="Transcribe">
    </form>
    {% if transcript %}
        <h3>Transcription:</h3>
        <p>{{ transcript }}</p>
    {% endif %}
</body>
</html>
"""

def transcribe_audio(audio_path):
    speech, sr = librosa.load(audio_path, sr=16000)
    input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values
    input_values = input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

@app.route("/", methods=["GET"])
def index():
    return render_template_string(html_template)

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio_file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["audio_file"]
    file_path = f"uploaded_audio.wav"
    file.save(file_path)

    try:
        transcript = transcribe_audio(file_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return render_template_string(html_template, transcript=transcript)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
