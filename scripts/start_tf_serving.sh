docker run -p 8501:8501 \
  --mount type=bind,source=$MODEL_SOURCE,target=$MODEL_TARGET \
  -e MODEL_NAME=$MODEL_NAME \
  -t tensorflow/serving