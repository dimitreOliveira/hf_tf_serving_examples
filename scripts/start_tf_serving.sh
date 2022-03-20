# torchserve --start \
# --ncs \
# --model-store model-store \
# --models fastrcnn=fastrcnn.mar

# docker run -p 8501:8501 \
#   --mount type=bind,source=$MODEL_SOURCE,target=$MODEL_TARGET \
#   -e MODEL_NAME=$MODEL_NAME -t tensorflow/serving



docker run -p 8501:8501 \
  --mount type=bind,source=$(pwd)/models/multi-label/1,target=/models/multi-label/1 \
  -e MODEL_NAME=multi-label -t tensorflow/serving