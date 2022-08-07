import os
import requests
import transformers
import gradio as gr
import numpy as np


server_port = os.environ.get("SERVER_PORT", 7861)
server_name = os.environ.get("SERVER_NAME", "0.0.0.0")
tokenizer_path = os.environ.get(
    "TOKENIZER_PATH", "./tokenizers/distilbert-base-uncased"
)
rest_url = os.environ.get(
    "TF_URL", "http://localhost:8501/v1/models/multiple_choice:predict"
)

print(f'Requesting predictions from: "{rest_url}"')
print(f'Loading tokenizer from: "{tokenizer_path}"')
tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)


def preprocess(
    prompt: str,
    choice1: str,
    choice2: str,
):
    return tokenizer([prompt, prompt], [choice1, choice2], padding=True)


def predict(prompt: str, choice1: str, choice2: str):
    tokenized_inputs = preprocess(prompt, choice1, choice2)
    batched_input = [dict(tokenized_inputs)]

    json_data = {"signature_name": "serving_default", "instances": batched_input}
    resp = requests.post(rest_url, json=json_data).json()
    prediction = resp["predictions"][0]
    return prediction["label"], prediction["probs"]


iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.inputs.Textbox(label="Prompt", placeholder="Input text prompt here..."),
        gr.inputs.Textbox(label="Choice 1", placeholder="Input text choice 1 here..."),
        gr.inputs.Textbox(label="Choice 2", placeholder="Input text choice 2 here..."),
    ],
    outputs=[gr.Textbox(label="Labels"), gr.Textbox(label="Probs")],
)
if __name__ == "__main__":
    app, local_url, share_url = iface.launch(
        server_port=server_port,
        server_name=server_name,
    )
