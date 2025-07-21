from faster_whisper import WhisperModel
import gradio as gr

# Load the model (use "large-v3" or "base" for faster CPU usage)
model = WhisperModel("base", device="cpu", compute_type="int8")

def transcribe(audio_file):
    segments, info = model.transcribe(audio_file)

    transcription = ""
    for segment in segments:
        transcription += segment.text + " "
    
    return transcription.strip()

# Gradio UI
gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="Upload Audio or Record"),
    outputs="text",
    title="Local Speech-to-Text (Whisper)",
    description="Transcribe audio locally using faster-whisper. No API needed."
).launch()
