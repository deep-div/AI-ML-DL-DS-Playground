import os
import time
import uuid
from datetime import datetime

import gradio as gr
import soundfile as sf

from model import get_pretrained_model, language_to_models


def MyPrint(s):
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    print(f"{date_time}: {s}")


title = "# Multilingual Text-to-speech (TTS)"

# css style is copied from
# https://huggingface.co/spaces/alphacep/asr/blob/main/app.py#L113
css = """
.result {display:flex;flex-direction:column}
.result_item {padding:15px;margin-bottom:8px;border-radius:15px;width:100%}
.result_item_success {background-color:mediumaquamarine;color:white;align-self:start}
.result_item_error {background-color:#ff7070;color:white;align-self:start}
"""

examples = [
    [
        "English",
        "csukuangfj/vits-piper-en_US-ryan-medium|1 speaker",
        "Welcome to the next-generation Kaldi Text-to-Speech demo, running entirely on CPU.",
        0,
        1.0,
    ],
    [
        "English",
        "csukuangfj/vits-piper-en_GB-southern_english_male-medium|8 speakers",
        "Machine learning and artificial intelligence are revolutionizing the tech industry.",
        0,
        1.0,
    ],
    [
        "English",
        "csukuangfj/vits-coqui-en-vctk|109 speakers",
        "The quick brown fox jumps over the lazy dog. It's a common pangram in English.",
        0,
        1.0,
    ],
    [
        "English",
        "csukuangfj/vits-piper-en_US-amy-medium|1 speaker",
        "On July 4th, 2025, we will celebrate with fireworks and music across the nation.",
        0,
        1.0,
    ],
    [
        "English",
        "csukuangfj/vits-piper-en_GB-alan-medium|1 speaker",
        "Please call 911 for emergencies. Your appointment is confirmed for September 1st.",
        0,
        1.0,
    ],
]




def update_model_dropdown(language: str):
    if language in language_to_models:
        choices = language_to_models[language]
        return gr.Dropdown(
            choices=choices,
            value=choices[0],
            interactive=True,
        )

    raise ValueError(f"Unsupported language: {language}")


def build_html_output(s: str, style: str = "result_item_success"):
    return f"""
    <div class='result'>
        <div class='result_item {style}'>
          {s}
        </div>
    </div>
    """


def process(language: str, repo_id: str, text: str, sid: str, speed: float):
    MyPrint(f"Input text: {text}. sid: {sid}, speed: {speed}")
    sid = int(sid)
    tts = get_pretrained_model(repo_id, speed)

    start = time.time()
    audio = tts.generate(text, sid=sid)
    end = time.time()

    if len(audio.samples) == 0:
        raise ValueError(
            "Error in generating audios. Please read previous error messages."
        )

    duration = len(audio.samples) / audio.sample_rate

    elapsed_seconds = end - start
    rtf = elapsed_seconds / duration

    info = f"""
    Wave duration  : {duration:.3f} s <br/>
    Processing time: {elapsed_seconds:.3f} s <br/>
    RTF: {elapsed_seconds:.3f}/{duration:.3f} = {rtf:.3f} <br/>
    """

    MyPrint(info)
    MyPrint(f"\nrepo_id: {repo_id}\ntext: {text}\nsid: {sid}\nspeed: {speed}")

    filename = str(uuid.uuid4())
    filename = f"{filename}.wav"
    sf.write(
        filename,
        audio.samples,
        samplerate=audio.sample_rate,
        subtype="PCM_16",
    )

    return filename, build_html_output(info)


demo = gr.Blocks(css=css)


with demo:
    gr.Markdown(title)
    language_choices = list(language_to_models.keys())

    language_radio = gr.Radio(
        label="Language",
        choices=language_choices,
        value=language_choices[0],
    )

    model_dropdown = gr.Dropdown(
        choices=language_to_models[language_choices[0]],
        label="Select a model",
        value=language_to_models[language_choices[0]][2],  
    )


    language_radio.change(
        update_model_dropdown,
        inputs=language_radio,
        outputs=model_dropdown,
    )

    with gr.Tabs():
        with gr.TabItem("Please input your text"):
            input_text = gr.Textbox(
                label="Input text",
                info="Your text",
                lines=3,
                placeholder="Please input your text here",
            )

            input_sid = gr.Textbox(
                label="Speaker ID",
                info="Speaker ID",
                lines=1,
                max_lines=1,
                value="0",
                placeholder="Speaker ID. Valid only for mult-speaker model",
            )

            input_speed = gr.Slider(
                minimum=0.1,
                maximum=10,
                value=1,
                step=0.1,
                label="Speed (larger->faster; smaller->slower)",
            )

            input_button = gr.Button("Submit")

            output_audio = gr.Audio(label="Output")

            output_info = gr.HTML(label="Info")

            gr.Examples(
                examples=examples,
                inputs=[
                    language_radio,
                    model_dropdown,
                    input_text,
                    input_sid,
                    input_speed,
                ],
                outputs=None,  # Do not auto-run on selection
                label="Click on an example to load it into the input fields. Then press Submit."
            )



        input_button.click(
            process,
            inputs=[
                language_radio,
                model_dropdown,
                input_text,
                input_sid,
                input_speed,
            ],
            outputs=[
                output_audio,
                output_info,
            ],
        )


def download_espeak_ng_data():
    os.system(
        """
    cd /tmp
    wget -qq https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/espeak-ng-data.tar.bz2
    tar xf espeak-ng-data.tar.bz2
    """
    )


if __name__ == "__main__":
    download_espeak_ng_data()
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    demo.launch()
