import fire
import transformers
import tensorflow as tf


def get_distilbert_multiclass(tokenizer_output: str="./tokenizers", model_output: str="./models/multi-class", 
                model_version: str="1", model_name: str=None, 
                tokenizer_name: str="distilbert-base-uncased"):
    # Tokenizer
    tokenizer = transformers.DistilBertTokenizer.from_pretrained(tokenizer_name)
    print(f"\nSaving tokenizer at {tokenizer_output}/{tokenizer_name}...")
    tokenizer.save_pretrained(f"{tokenizer_output}/{tokenizer_name}")

    # Model
    if model_name:
        print(f"Loading the model from pre-trained checkpoint {model_name}...")        
        base_model = transformers.TFDistilBertModel.from_pretrained(model_name)
    else:
        print("Loading the model from scratch...")
        config = transformers.DistilBertConfig()
        base_model = transformers.TFDistilBertModel(config)

    # Model architecture
    inputs = {layer: tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name=layer) for layer in tokenizer.model_input_names}

    encoded = base_model(inputs)["last_hidden_state"]
    cls_token = encoded[:, 0, :]

    outputs = tf.keras.layers.Dense(5, activation="softmax")(cls_token)
    model = tf.keras.models.Model(inputs=inputs.values(), outputs=outputs)

    print("\nCreated model architecture:")
    print(model.summary())
    print(f"Input config: {model.input}")

    print(f"\nSaving model at {model_output}...")
    model.save(f"{model_output}/{model_version}")


def get_distilbert_multilabel(tokenizer_output: str="./tokenizers", model_output: str="./models/multi-label", 
                model_version: str="1", model_name: str=None, 
                tokenizer_name: str="distilbert-base-uncased"):
    # Tokenizer
    tokenizer = transformers.DistilBertTokenizer.from_pretrained(tokenizer_name)
    print(f"\nSaving tokenizer at {tokenizer_output}/{tokenizer_name}...")
    tokenizer.save_pretrained(f"{tokenizer_output}/{tokenizer_name}")

    # Model
    if model_name:
        print(f"Loading the model from pre-trained checkpoint {model_name}...")        
        base_model = transformers.TFDistilBertModel.from_pretrained(model_name)
    else:
        print("Loading the model from scratch...")
        config = transformers.DistilBertConfig()
        base_model = transformers.TFDistilBertModel(config)

    # Model architecture
    inputs = {layer: tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name=layer) for layer in tokenizer.model_input_names}

    encoded = base_model(inputs)["last_hidden_state"]
    cls_token = encoded[:, 0, :]

    outputs = tf.keras.layers.Dense(5, activation="sigmoid")(cls_token)
    model = tf.keras.models.Model(inputs=inputs.values(), outputs=outputs)

    print("\nCreated model architecture:")
    print(model.summary())
    print(f"Input config: {model.input}")

    print(f"\nSaving model at {model_output}...")
    model.save(f"{model_output}/{model_version}")


def get_distilbert_regression(tokenizer_output: str="./tokenizers", model_output: str="./models/regression", 
                model_version: str="1", model_name: str=None, 
                tokenizer_name: str="distilbert-base-uncased"):
    # Tokenizer
    tokenizer = transformers.DistilBertTokenizer.from_pretrained(tokenizer_name)
    print(f"\nSaving tokenizer at {tokenizer_output}/{tokenizer_name}...")
    tokenizer.save_pretrained(f"{tokenizer_output}/{tokenizer_name}")

    # Model
    if model_name:
        print(f"Loading the model from pre-trained checkpoint {model_name}...")        
        base_model = transformers.TFDistilBertModel.from_pretrained(model_name)
    else:
        print("Loading the model from scratch...")
        config = transformers.DistilBertConfig()
        base_model = transformers.TFDistilBertModel(config)

    # Model architecture
    inputs = {layer: tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name=layer) for layer in tokenizer.model_input_names}

    encoded = base_model(inputs)["last_hidden_state"]
    cls_token = encoded[:, 0, :]

    outputs = tf.keras.layers.Dense(1, activation="linear")(cls_token)
    model = tf.keras.models.Model(inputs=inputs.values(), outputs=outputs)

    print("\nCreated model architecture:")
    print(model.summary())
    print(f"Input config: {model.input}")

    print(f"\nSaving model at {model_output}...")
    model.save(f"{model_output}/{model_version}")


def get_distilbert_embedding(tokenizer_output: str="./tokenizers", model_output: str="./models/embedding", 
                model_version: str="1", model_name: str=None, 
                tokenizer_name: str="distilbert-base-uncased"):
    # Tokenizer
    tokenizer = transformers.DistilBertTokenizer.from_pretrained(tokenizer_name)
    print(f"\nSaving tokenizer at {tokenizer_output}/{tokenizer_name}...")
    tokenizer.save_pretrained(f"{tokenizer_output}/{tokenizer_name}")

    # Model
    if model_name:
        print(f"Loading the model from pre-trained checkpoint {model_name}...")        
        base_model = transformers.TFDistilBertModel.from_pretrained(model_name)
    else:
        print("Loading the model from scratch...")
        config = transformers.DistilBertConfig()
        base_model = transformers.TFDistilBertModel(config)

    # Model architecture
    inputs = {layer: tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name=layer) for layer in tokenizer.model_input_names}

    encoded = base_model(inputs)["last_hidden_state"]

    outputs = tf.keras.layers.GlobalAveragePooling1D()(encoded)
    model = tf.keras.models.Model(inputs=inputs.values(), outputs=outputs)

    print("\nCreated model architecture:")
    print(model.summary())
    print(f"Input config: {model.input}")

    print(f"\nSaving model at {model_output}...")
    model.save(f"{model_output}/{model_version}")


if __name__ == "__main__":
    fire.Fire()
