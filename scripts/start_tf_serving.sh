MODEL_SOURCE=$(pwd)/models/multi-label/1
MODEL_TARGET=/models/multi-label/1
MODEL_NAME=multi-label

docker run -p 8501:8501 \
  --mount type=bind,source=$MODEL_SOURCE,target=$MODEL_TARGET \
  -e MODEL_NAME=$MODEL_NAME \
  -t tensorflow/serving