import os
import requests
import transformers
import gradio as gr


server_port = os.environ.get("SERVER_PORT", 7861)
server_name = os.environ.get("SERVER_NAME", "0.0.0.0")
tokenizer_path = os.environ.get("TOKENIZER_PATH", "./tokenizers/distilbert-base-uncased")
rest_url = os.environ.get("TF_URL", "http://localhost:8501/v1/models/multi-class:predict")

print(f"Requesting predictions from: {rest_url}")
print(f"Loading tokenizer from: {tokenizer_path}")
tokenizer = transformers.DistilBertTokenizer.from_pretrained(tokenizer_path)

def preprocess(name, tokenizer):
    return tokenizer(name)

def predict(input):
    batched_input = [dict(preprocess(input, tokenizer))]

    json_data = {
        "signature_name": "serving_default", 
        "instances": batched_input
    }
    resp = requests.post(rest_url, json=json_data).json()
    prediction = resp["predictions"][0]
    return prediction

iface = gr.Interface(
    fn=predict,
    inputs=gr.inputs.Textbox(placeholder="Input text here..."),
    outputs="text",
)
if __name__ == "__main__":
    app, local_url, share_url = iface.launch(server_port=server_port, server_name=server_name, )
