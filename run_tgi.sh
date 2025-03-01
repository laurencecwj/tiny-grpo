# docker pull ghcr.io/huggingface/text-generation-inference:3.1.0

volume=`pwd`/data
# docker run --gpus all --shm-size 1g -p 7777:80 -v $volume:/data -d ghcr.io/huggingface/text-generation-inference:3.1.0 --model-id /data/unsloth/Llama-3.2-1B-Instruct
 docker run --gpus all --shm-size 1g -p 7777:80 -v $volume:/data -d ghcr.io/huggingface/text-generation-inference:3.1.0 --model-id /data/my