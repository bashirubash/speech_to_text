import gradio as gr
from faster_whisper import WhisperModel
import os

# Load Whisper tiny or base model depending on your accuracy/speed needs
model_size = "tiny"  # or "base" for better accuracy (still Render compatible)
model = WhisperModel(model_size, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="int8")

def transcribe(audio_path):
    if audio_path is None or not os.path.exists(audio_path):
        return "Please upload or record an audio file."

    segments, info = model.transcribe(audio_path, beam_size=5)  # beam search for better accuracy

    transcription = " ".join(segment.text.strip() for segment in segments)
    return transcription.strip()

iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="Record or Upload Audio"),
    outputs="text",
    title="AI-Powered Speech-to-Text Transcription",
    description="Smart, fast, and accurate speech-to-text using Whisper Tiny or Base (Render compatible)."
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    iface.launch(server_name="0.0.0.0", server_port=port)
