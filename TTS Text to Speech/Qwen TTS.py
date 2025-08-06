import os
import requests
import tempfile

import dashscope
import gradio as gr

# Load your API key from environment variable or replace with your actual key
API_KEY = os.environ['API_KEY']

def tts_gradio(text: str, voice: str) -> str:
    """
    Call Qwen-TTS API to synthesize speech, save the audio to a temp file,
    and return the file path for Gradio to play.
    """
    response = dashscope.audio.qwen_tts.SpeechSynthesizer.call(
        model="qwen-tts-latest",
        api_key=API_KEY,
        text=text,
        voice=voice,
    )
    audio_url = response.output.audio["url"]

    # Download the audio
    resp = requests.get(audio_url)
    resp.raise_for_status()

    # Save to a temporary .wav file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(resp.content)
    tmp.flush()
    tmp.close()

    return tmp.name

# Gradio Interface
demo = gr.Interface(
    fn=tts_gradio,
    inputs=[
        gr.Textbox(lines=4, label="Input Text"),
        gr.Dropdown(
            choices=["Dylan", "Sunny", "Jada", "Cherry", "Ethan", "Serena", "Chelsie"],
            value="Dylan",
            label="Speaker Voice"
        ),
    ],
    outputs=gr.Audio(label="Synthesized Audio"),
    title="Qwen-TTS Gradio Demo",
    description="Enter your text, select a speaker voice, and click 'Submit' to hear the speech synthesis.",
    allow_flagging="never",
    examples=[
        ["Hello! Welcome to the Qwen-TTS demo. This is a sample speech synthesis.", "Dylan"],
        ["Today is a beautiful day. The sun is shining and the sky is clear.", "Sunny"],
        ["Artificial Intelligence is transforming the world. Stay curious!", "Cherry"],
        ["Thank you for using this demo. Have a great day!", "Serena"],
        ["This voice sounds so natural, doesn’t it? Let’s test another speaker.", "Ethan"],
    ]
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
