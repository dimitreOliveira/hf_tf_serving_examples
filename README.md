# Simple examples of serving HuggingFace models with TensorFlow Serving.

# Repository content
- [Setup](#setup)
- [Download sample model](#download-sample-model)
- [Start TF Serving](#start-tf-serving)
- [Inference](#inference)
- [References](#references)

## Setup
- Docker - [Docker installation guide](https://help.github.com/en/github/getting-started-with-github/set-up-git)
- TensorFlow - [TensorFlow installation guide](https://www.tensorflow.org/install)
- TensorFlow Serving - [TensorFlow Serving installation guide](https://www.tensorflow.org/tfx/serving/docker)
- HuggingFace - [HuggingFace installation guide](https://huggingface.co/docs/transformers/installation)

## Download sample model
_*this step is optional_

**Available models:**
- DistilBERT (multi-label)
```bash
python3 sample_models/tf_distilbert_multilabel.py get_model
```

## Start TF Serving
*_requires Docker_

*_parameters refer to "DistilBERT (multi-label)" example_

```bash
MODEL_SOURCE=$(pwd)/models/multi-label/1 MODEL_TARGET=/models/multi-label/1 MODEL_NAME=multi-label sh scripts/start_tf_serving.sh
```

## Inference
_*bash command not working yet, needs integration with tokenizer_
```bash
curl -d '{"instances": sample text}' -X POST http://localhost:8501/v1/models/multi-label:predict
```
Or use the notebook  at `notebooks/inference.ipynb`

## References
- [TensorFlow Serving with Docker](https://www.tensorflow.org/tfx/serving/docker#creating_your_own_serving_image)