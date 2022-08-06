import tensorflow as tf


def seq_classification_signature(
    model: tf.keras.Model, labels: list, input_names: list
):
    input_signature = [
        tf.TensorSpec([None], tf.int32, name=name) for name in input_names
    ]

    @tf.function(input_signature=[input_signature])
    def serving_fn(input):
        predictions = model(input).logits
        pred_source = tf.gather(
            params=tf.constant(labels, dtype=tf.string),
            indices=tf.argmax(predictions, axis=1),
        )
        probs = tf.nn.softmax(predictions, axis=1)
        probs = tf.reduce_max(probs, axis=1)
        return {"label": pred_source, "probs": probs}

    return serving_fn


def token_classification_signature(
    model: tf.keras.Model, labels: list, input_names: list
):
    input_signature = [
        tf.TensorSpec([None], tf.int32, name=name) for name in input_names
    ]

    @tf.function(input_signature=[input_signature])
    def serving_fn(input):
        predictions = model(input).logits
        tokens_classes = tf.gather(
            params=tf.constant(labels, dtype=tf.string),
            indices=tf.argmax(predictions, axis=-1),
        )
        probs = tf.reduce_max(predictions, axis=-1)
        return {"label": tokens_classes, "probs": probs}

    return serving_fn


def multiple_choice_signature(model: tf.keras.Model, input_names: list):
    input_signature = [
        tf.TensorSpec([None, None, None], tf.int32, name=name) for name in input_names
    ]

    @tf.function(input_signature=[input_signature])
    def serving_fn(input):
        predictions = model(input).logits
        probs = tf.nn.softmax(predictions, axis=1)
        choice_probs = tf.reduce_max(probs, axis=-1)
        choice = tf.argmax(probs, axis=-1)
        return {"label": choice, "probs": choice_probs}

    return serving_fn


def qa_signature(model: tf.keras.Model, input_names: list):
    input_signature = [
        tf.TensorSpec([None], tf.int32, name=name) for name in input_names
    ]

    @tf.function(input_signature=[input_signature])
    def serving_fn(input):
        predictions = model(input)
        start = tf.math.argmax(predictions.start_logits, axis=-1)
        end = tf.math.argmax(predictions.end_logits, axis=-1)
        start_probs = tf.nn.softmax(predictions.start_logits, axis=1)
        start_probs = tf.reduce_max(start_probs, axis=1)
        end_probs = tf.nn.softmax(predictions.end_logits, axis=1)
        end_probs = tf.reduce_max(end_probs, axis=1)
        return {
            "start": start,
            "end": end,
            "start_probs": start_probs,
            "end_probs": end_probs,
        }

    return serving_fn
