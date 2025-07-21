from faster_whisper import WhisperModel
import gradio as gr
import os

model = WhisperModel("tiny", device="cpu", compute_type="int8")

def transcribe(audio):
    if audio is None:
        return "Please upload or record an audio file."

    segments, info = model.transcribe(audio)

    transcription = ""
    for segment in segments:
        transcription += segment.text + " "

    return transcription.strip()

iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="Record or Upload Audio"),
    outputs="text",
    title="Tiny Whisper Speech-to-Text",
    description="Transcribe audio to text using faster-whisper tiny model (Render compatible)."
)

# Use dynamic port from Render
port = int(os.environ.get("PORT", 7860))

iface.launch(server_name="0.0.0.0", server_port=port)
