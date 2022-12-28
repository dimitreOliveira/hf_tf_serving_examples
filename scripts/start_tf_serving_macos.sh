docker run -p 8501:8501 \
  -p 8500:8500 \
  --mount type=bind,source=$MODEL_SOURCE,target=$MODEL_TARGET \
  -e MODEL_NAME=$MODEL_NAME \
  -t emacski/tensorflow-serving:latest-linux_arm64