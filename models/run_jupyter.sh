docker build -t jupyter_image .

docker run --rm -it \
    --device=nvidia.com/gpu=all \
    -p 8888:8888 \
    --volume .:/home/jovyan/ \
    jupyter_image