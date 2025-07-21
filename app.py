from faster_whisper import WhisperModel
import gradio as gr
import os

# Load the tiny model in int8 for CPU-friendly usage
model = WhisperModel("tiny", device="cpu", compute_type="int8")

def transcribe(audio_path):
    if audio_path is None or not os.path.exists(audio_path):
        return "Please upload or record an audio file."

    segments, info = model.transcribe(audio_path)

    transcription = ""
    for segment in segments:
        transcription += segment.text.strip() + " "

    return transcription.strip()

iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="Record or Upload Audio"),
    outputs="text",
    title="Tiny Whisper Speech-to-Text",
    description="Transcribe audio using faster-whisper tiny model. Fully Render compatible."
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    iface.launch(server_name="0.0.0.0", server_port=port)
