import gradio as gr
from faster_whisper import WhisperModel
import torch
import os

# Automatically select GPU if available (Render will likely run CPU, but this handles both)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Whisper tiny model
model = WhisperModel("tiny", device=device, compute_type="int8" if device == "cpu" else "float16")

def transcribe(audio_path):
    if audio_path is None:
        return "Please upload or record an audio file."

    segments, _ = model.transcribe(audio_path, beam_size=5)  # Beam search for better accuracy

    transcription = " ".join(segment.text.strip() for segment in segments)
    return transcription

# Create Gradio interface
iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="Record or Upload Audio"),
    outputs="text",
    title="Smart AI Speech-to-Text (Tiny Whisper)",
    description="Fast and accurate transcription using faster-whisper tiny model. Works on CPU and GPU. Render compatible."
)

# Use Render's dynamic port binding
port = int(os.environ.get("PORT", 7860))
iface.launch(server_name="0.0.0.0", server_port=port)
