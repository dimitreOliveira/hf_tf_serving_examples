import tensorflow as tf


def seq_classification_signature(
    model: tf.keras.Model, labels: list, input_names: list
):
    input_signature = [
        tf.TensorSpec([None, None], tf.int32, name=name) for name in input_names
    ]

    @tf.function(input_signature=[input_signature])
    def serving_fn(input):
        predictions = model(input).logits
        pred_source = tf.gather(
            params=tf.constant(labels, dtype=tf.string),
            indices=tf.argmax(predictions, axis=-1),
        )
        probs = tf.nn.softmax(predictions, axis=-1)
        probs = tf.reduce_max(probs, axis=-1)
        return {
            "label": tf.expand_dims(pred_source, axis=-1),
            "probs": tf.expand_dims(probs, axis=-1),
        }

    return serving_fn


def token_classification_signature(
    model: tf.keras.Model, labels: list, input_names: list
):
    input_signature = [
        tf.TensorSpec([None, None], tf.int32, name=name) for name in input_names
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
        return {
            "label": tf.expand_dims(choice, axis=-1),
            "probs": tf.expand_dims(choice_probs, axis=-1),
        }

    return serving_fn


def qa_signature(model: tf.keras.Model, input_names: list):
    input_signature = [
        tf.TensorSpec([None, None], tf.int32, name=name) for name in input_names
    ]

    @tf.function(input_signature=[input_signature])
    def serving_fn(input):
        predictions = model(input)
        start = tf.math.argmax(predictions.start_logits, axis=-1)
        end = tf.math.argmax(predictions.end_logits, axis=-1)
        start_probs = tf.nn.softmax(predictions.start_logits, axis=-1)
        start_probs = tf.reduce_max(start_probs, axis=-1)
        end_probs = tf.nn.softmax(predictions.end_logits, axis=-1)
        end_probs = tf.reduce_max(end_probs, axis=1)
        return {
            "start": tf.expand_dims(start, axis=-1),
            "end": tf.expand_dims(end, axis=-1),
            "start_probs": tf.expand_dims(start_probs, axis=-1),
            "end_probs": tf.expand_dims(end_probs, axis=-1),
        }

    return serving_fn


def text_generation_signature(model: tf.keras.Model, eos_token: int, max_len: int):
    @tf.function(input_signature=[tf.TensorSpec([None], tf.int32)])
    def serving_fn(input):
        def cond(inputs):
            length_cond = tf.less(tf.size(inputs), max_len)
            eos_cond = tf.math.not_equal(
                tf.gather(inputs, (tf.size(inputs) - 1), axis=-1),
                tf.constant(eos_token),
            )
            return tf.math.logical_and(length_cond, eos_cond)

        def body(inputs):
            predictions = model(inputs)
            next_token_logits = predictions.logits[-1, :]
            next_token_scores = tf.nn.log_softmax(next_token_logits, axis=-1)
            next_token_id = tf.argmax(next_token_scores, axis=-1)
            next_token_id = tf.cast(
                tf.expand_dims(next_token_id, axis=0), dtype="int32"
            )
            return (tf.concat([inputs, next_token_id], axis=-1),)

        generated = tf.while_loop(cond=cond, body=body, loop_vars=[input])
        generated = tf.convert_to_tensor(generated)
        return {"generated": generated}

    return serving_fn
