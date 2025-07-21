import os
import openai
import gradio as gr

# Load OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def transcribe(audio_path):
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
        return transcript["text"]
    except Exception as e:
        return f"Error: {str(e)}"

iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="Upload or Record Audio"),
    outputs=gr.Textbox(label="Transcription"),
    title="Speech to Text using ChatGPT API (Whisper)",
    description="Upload or record audio and get transcription using OpenAI Whisper API."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 8080)))
