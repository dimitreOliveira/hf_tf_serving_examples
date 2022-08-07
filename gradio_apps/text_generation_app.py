import os
import requests
import transformers
import gradio as gr


server_port = os.environ.get("SERVER_PORT", 7861)
server_name = os.environ.get("SERVER_NAME", "0.0.0.0")
tokenizer_path = os.environ.get("TOKENIZER_PATH", "./tokenizers/distilgpt2")
rest_url = os.environ.get(
    "TF_URL", "http://localhost:8501/v1/models/text_generation:predict"
)

print(f'Requesting predictions from: "{rest_url}"')
print(f'Loading tokenizer from: "{tokenizer_path}"')
tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)


def preprocess(text: str):
    return tokenizer(text).input_ids


def postprocess(input_ids: list):
    return tokenizer.decode(input_ids, skip_special_tokens=True)


def predict(input: str):
    token_text = preprocess(input)
    json_data = {"signature_name": "serving_default", "instances": token_text}
    resp = requests.post(rest_url, json=json_data).json()
    prediction = resp["predictions"][0]
    generated = postprocess(prediction)
    return generated


iface = gr.Interface(
    fn=predict,
    inputs=gr.inputs.Textbox(placeholder="Input text here..."),
    outputs=gr.Textbox(label="Generated"),
)
if __name__ == "__main__":
    app, local_url, share_url = iface.launch(
        server_port=server_port,
        server_name=server_name,
    )
