# Simple examples of serving HuggingFace models with TensorFlow Serving.

# Repository content
- [Setup](#setup)
- [Start TensorFlow Serving](#start-tensorflow-serving)
- [Inference](#inference)
- [References](#references)

## Setup
- Docker - [Docker installation guide](https://help.github.com/en/github/getting-started-with-github/set-up-git)
- TensorFlow - [TensorFlow installation guide](https://www.tensorflow.org/install)
- TensorFlow Serving - [TensorFlow Serving installation guide](https://www.tensorflow.org/tfx/serving/docker)
- HuggingFace - [HuggingFace installation guide](https://huggingface.co/docs/transformers/installation)

## Start TensorFlow Serving
*_requires Docker_

*_parameters refer to "DistilBERT (embedding)" sample example_

```bash
MODEL_SOURCE=$(pwd)/models/embedding/saved_model/1 MODEL_TARGET=/models/embedding/1 MODEL_NAME=embedding sh scripts/start_tf_serving.sh
```

*_parameters refer to "DistilBERT (custom)" sample example_

```bash
MODEL_SOURCE=$(pwd)/models/custom/1 MODEL_TARGET=/models/custom/1 MODEL_NAME=custom sh scripts/start_tf_serving.sh
```
Parameters:
- `MODEL_SOURCE`: path to the model in your local system.
- `MODEL_TARGET`: path to the model in the Docker env.
- `MODEL_NAME`: Model name used by TFServing, this name will be part of the API URL.

_After finished you can use `docker ps` to check active containers and then `docker stop` to stop it._

If you don't have a model to use, you can create one using one of the sample models:

### Available sample models:
- DistilBERT (embedding)
```bash
python sample_models/text_models.py get_distilbert_embedding
```
- DistilBERT (sequence classification)
```bash
python sample_models/text_models.py get_distilbert_sequence_classification
```
- DistilBERT (token classification - NER)
```bash
python sample_models/text_models.py get_distilbert_token_classification
```
- DistilBERT (multiple choice)
```bash
python sample_models/text_models.py get_distilbert_multiple_choice
```
- DistilBERT (question answering)
```bash
python sample_models/text_models.py get_distilbert_qa
```
- DistilBERT (custom)
```bash
python sample_models/text_models.py get_distilbert_custom
```

## Inference
We have two options to access the model and make inferences.

### Notebook
 - Just use the notebook at `notebooks/text_inference.ipynb`

### Gradio APP
- Run the `app.py` command folder for your specific use case at the `gradio_apps`
- Available use cases:
  - Text:
    - Generic
      ```bash
      TF_URL="http://localhost:8501/v1/models/embedding:predict" TOKENIZER_PATH="./tokenizers/distilbert-base-uncased" python gradio_apps/text_app.py
      ```
    - token classification - NER
      ```bash
      TF_URL="http://localhost:8501/v1/models/token_classification:predict" TOKENIZER_PATH="./tokenizers/distilbert-base-uncased" python gradio_apps/text_ner_app.py
      ```
    - multiple choice
      ```bash
      TF_URL="http://localhost:8501/v1/models/multiple_choice:predict" TOKENIZER_PATH="./tokenizers/distilbert-base-uncased" python gradio_apps/text_multiple_choice_app.py
      ```
    - question answering
      ```bash
      TF_URL="http://localhost:8501/v1/models/qa:predict" TOKENIZER_PATH="./tokenizers/distilbert-base-uncased" python gradio_apps/text_qa_app.py
      ```

 *_ To be more generic, predictions from the Gradio apps will return raw outputs_
 *_Gradio apps requires you to define environment variables_
 
 #### For all use cases:
 - `TF_URL`: REST API URL provided by your TF Serving.
   - e.g. `"http://localhost:8501/v1/models/embedding:predict"`
     - Swap {`embedding`} with your model's name
 
 #### Text use case:
 - `TOKENIZER_PATH`: path to the tokenizer in your local system.
   - e.g. `"./tokenizerstokenizers/distilbert-base-uncased"`

## References
- [TensorFlow Serving with Docker](https://www.tensorflow.org/tfx/serving/docker#creating_your_own_serving_image)
- [Gradio - Getting Started](https://gradio.app/getting_started/)
- [Deploying TensorFlow Vision Models in Hugging Face with TF Serving](https://huggingface.co/blog/tf-serving-vision)