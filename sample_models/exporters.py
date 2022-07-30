import tensorflow as tf


def seq_classification_exporter(model, labels, tokenizer):
    input_signature = [tf.TensorSpec([None], tf.int32, name=name) for name in tokenizer.model_input_names]
    @tf.function(input_signature=[input_signature])
    def serving_fn(input):
        predictions = model(input).logits
        pred_source = tf.gather(params=tf.constant(labels, dtype=tf.string), 
                                indices=tf.argmax(predictions, axis=1))
        probs = tf.nn.softmax(predictions, axis=1)
        probs = tf.reduce_max(probs, axis=1)
        return {"label": pred_source, "probs": probs}

    return serving_fn


def token_classification_exporter(model, labels, tokenizer):
    input_signature = [tf.TensorSpec([None], tf.int32, name=name) for name in tokenizer.model_input_names]
    @tf.function(input_signature=[input_signature])
    def serving_fn(input):
        predictions = model(input).logits
        tokens_classes = tf.gather(params=tf.constant(labels, dtype=tf.string), 
                                   indices=tf.argmax(predictions, axis=-1))
        probs = tf.reduce_max(predictions, axis=-1)
        return {"label": tokens_classes, "probs": probs}

    return serving_fn
