import os
import requests
import transformers
import gradio as gr


server_port = os.environ.get("SERVER_PORT", 7861)
server_name = os.environ.get("SERVER_NAME", "0.0.0.0")
tokenizer_path = os.environ.get(
    "TOKENIZER_PATH", "./tokenizers/distilbert-base-uncased"
)
rest_url = os.environ.get("TF_URL", "http://localhost:8501/v1/models/qa:predict")

print(f'Requesting predictions from: "{rest_url}"')
print(f'Loading tokenizer from: "{tokenizer_path}"')
tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)


def preprocess(
    question: str,
    text: str,
    tokenizer: transformers.tokenization_utils.PreTrainedTokenizer,
):
    return tokenizer(question, text)


def postprocess(
    start: int,
    end: int,
    input_ids: list,
    tokenizer: transformers.tokenization_utils.PreTrainedTokenizer,
):
    answer_tokens = input_ids[start : (end + 1)]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    return answer


def predict(question, text):
    tokenized_inputs = preprocess(question, text, tokenizer)
    batched_input = [dict(tokenized_inputs)]

    json_data = {"signature_name": "serving_default", "instances": batched_input}
    resp = requests.post(rest_url, json=json_data).json()
    prediction = resp["predictions"][0]
    answer = postprocess(
        prediction["start"], prediction["end"], tokenized_inputs.input_ids, tokenizer
    )
    return answer


iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.inputs.Textbox(label="Question", placeholder="Input text question here..."),
        gr.inputs.Textbox(label="Text", placeholder="Input text here..."),
    ],
    outputs=gr.Textbox(label="Answer"),
)
if __name__ == "__main__":
    app, local_url, share_url = iface.launch(
        server_port=server_port,
        server_name=server_name,
    )
