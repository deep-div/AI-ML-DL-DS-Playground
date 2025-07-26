import gradio as gr
from google import genai
from google.genai import types
import wave
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

# Save audio from PCM to WAV
def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)

# Gemini TTS generation function
def generate_speech(text, voice):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice
                        )
                    )
                )
            )
        )

        audio_data = response.candidates[0].content.parts[0].inline_data.data
        output_path = "output.wav"
        wave_file(output_path, audio_data)
        return output_path, output_path, "Speech generated successfully."

    except Exception as e:
        return None, None, f"Error: {str(e)}"

# Gradio app using Blocks
with gr.Blocks(title="Gemini TTS Demo") as demo:
    gr.Markdown("## Google Gemini Text-to-Speech")
    gr.Markdown("Enter text below, choose a voice, and listen to the generated speech.")

    with gr.Row():
        text_input = gr.Textbox(
            lines=3,
            label="Enter Text",
            placeholder="Example: Welcome to the world of AI."
        )
        voice_input = gr.Dropdown(
            choices=["Kore", "Wes"],
            value="Kore",
            label="Select Voice"
        )

    with gr.Row():
        generate_btn = gr.Button("Generate Speech", variant="primary")

    with gr.Row():
        audio_output = gr.Audio(label="Generated Audio")
        file_output = gr.File(label="Download Audio File")
        status_output = gr.Textbox(label="Status", interactive=False)

    examples = gr.Examples(
        examples=[
            ["Good morning! Hope you have a great day ahead.", "Kore"],
            ["Welcome to the future of AI voice generation.", "Wes"],
            ["Your appointment is scheduled for 3 PM on Monday.", "Kore"],
            ["This is a demo of Google's Gemini text-to-speech feature.", "Wes"],
        ],
        inputs=[text_input, voice_input],
    )

    generate_btn.click(
        fn=generate_speech,
        inputs=[text_input, voice_input],
        outputs=[audio_output, file_output, status_output],
    )

demo.launch()