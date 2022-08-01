import fire
import transformers
import tensorflow as tf
import tensorflow.keras.layers as L
from signatures import seq_classification_signature, token_classification_signature, multiple_choice_signature, qa_signature


def get_distilbert_embedding(tokenizer_output: str="./tokenizers", model_output: str="./models/embedding", 
                              model_name: str="distilbert-base-uncased"):
    tokenizer = transformers.DistilBertTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(f"{tokenizer_output}/{model_name}")
    print(f"Saving tokenizer at {tokenizer_output}/{model_name}")

    print(f'Loading model from pre-trained checkpoint "{model_name}"')
    model = transformers.TFDistilBertModel.from_pretrained(model_name)
    print(model.summary())

    model.save_pretrained(model_output, saved_model=True)
    print(f"Saving model at {model_output}")


def get_distilbert_sequence_classification(tokenizer_output: str="./tokenizers", model_output: str="./models/seq_classification", 
                                            model_version: str="1", model_name: str="distilbert-base-uncased"):
    tokenizer = transformers.DistilBertTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(f"{tokenizer_output}/{model_name}")
    print(f"Saving tokenizer at {tokenizer_output}/{model_name}")

    print(f'Loading model from pre-trained checkpoint "{model_name}"')
    model = transformers.TFDistilBertForSequenceClassification.from_pretrained(model_name)
    print(model.summary())

    labels = list(model.config.id2label.values())
    input_names = tokenizer.model_input_names
    tf.saved_model.save(obj=model, 
                        export_dir=f"{model_output}/{model_version}", 
                        signatures={"serving_default": seq_classification_signature(model, labels, input_names)})
    print(f"Saving model at {model_output}/{model_version}")


def get_distilbert_token_classification(tokenizer_output: str="./tokenizers", model_output: str="./models/token_classification", 
                                        model_version: str="1", model_name: str="distilbert-base-uncased"):
    tokenizer = transformers.DistilBertTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(f"{tokenizer_output}/{model_name}")
    print(f"Saving tokenizer at {tokenizer_output}/{model_name}")

    print(f'Loading model from pre-trained checkpoint "{model_name}"')
    model = transformers.TFDistilBertForTokenClassification.from_pretrained(model_name)
    print(model.summary())

    labels = list(model.config.id2label.values())
    input_names = tokenizer.model_input_names
    tf.saved_model.save(obj=model, 
                        export_dir=f"{model_output}/{model_version}", 
                        signatures={"serving_default": token_classification_signature(model, labels, input_names)})
    print(f"Saving model at {model_output}/{model_version}")


def get_distilbert_multiple_choice(tokenizer_output: str="./tokenizers", model_output: str="./models/multiple_choice", 
                                    model_version: str="1", model_name: str="distilbert-base-uncased"):
    tokenizer = transformers.DistilBertTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(f"{tokenizer_output}/{model_name}")
    print(f"Saving tokenizer at {tokenizer_output}/{model_name}")

    print(f'Loading model from pre-trained checkpoint "{model_name}"')
    model = transformers.TFDistilBertForMultipleChoice.from_pretrained(model_name)
    print(model.summary())

    input_names = tokenizer.model_input_names
    tf.saved_model.save(obj=model, 
                        export_dir=f"{model_output}/{model_version}", 
                        signatures={"serving_default": multiple_choice_signature(model, input_names)})
    print(f"Saving model at {model_output}/{model_version}")


def get_distilbert_qa(tokenizer_output: str="./tokenizers", model_output: str="./models/qa", model_version: str="1", 
                        model_name: str="distilbert-base-uncased"):
    tokenizer = transformers.DistilBertTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(f"{tokenizer_output}/{model_name}")
    print(f"Saving tokenizer at {tokenizer_output}/{model_name}")

    print(f'Loading model from pre-trained checkpoint "{model_name}"')
    model = transformers.TFDistilBertForQuestionAnswering.from_pretrained(model_name)
    print(model.summary())

    input_names = tokenizer.model_input_names
    tf.saved_model.save(obj=model, 
                        export_dir=f"{model_output}/{model_version}", 
                        signatures={"serving_default": qa_signature(model, input_names)})
    print(f"Saving model at {model_output}/{model_version}")


def get_distilbert_custom(tokenizer_output: str="./tokenizers", model_output: str="./models/custom", 
                              model_version: str="1", model_name: str="distilbert-base-uncased"):
    tokenizer = transformers.DistilBertTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(f"{tokenizer_output}/{model_name}")
    print(f"Saving tokenizer at {tokenizer_output}/{model_name}")

    print(f'Loading model from pre-trained checkpoint "{model_name}"')    
    base_model = transformers.TFDistilBertModel.from_pretrained(model_name)

    # Model architecture
    inputs = {layer: L.Input(shape=(None,), dtype=tf.int32, name=layer) for layer in tokenizer.model_input_names}

    encoded = base_model(inputs)["last_hidden_state"]
    encoded = L.GlobalAveragePooling1D()(encoded)
    outputs = L.Dense(3, activation="softmax")(encoded)
    model = tf.keras.models.Model(inputs=inputs.values(), outputs=outputs)
    print(model.summary())

    tf.saved_model.save(obj=model, 
                        export_dir=f"{model_output}/{model_version}")
    print(f"Saving model at {model_output}/{model_version}")


if __name__ == "__main__":
    fire.Fire()
