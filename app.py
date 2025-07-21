import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import gradio as gr
import librosa

# Use Hugging Face pretrained model (English)
model_name = "facebook/wav2vec2-base-960h"

# Load processor and model
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Define the transcribe function
def transcribe(audio):
    sr = 16000  # Model expects 16kHz input
    audio, _ = librosa.load(audio, sr=sr)
    input_values = processor(audio, return_tensors="pt", sampling_rate=sr).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

# Create Gradio Interface
interface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(source="microphone", type="filepath", label="Record or Upload Audio"),
    outputs=gr.Textbox(label="Transcribed Text"),
    title="Real-Time Speech to Text",
    description="This app converts speech to text using Facebook's Wav2Vec2 model. Powered by Hugging Face Transformers and Gradio."
)

# Launch the app
interface.launch()
