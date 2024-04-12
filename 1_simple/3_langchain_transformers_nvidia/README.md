# Overview
Example chatbot using langchain and transformer with Nvidia support


# Running
1. Run `docker compose build`
2. Run `docker compose up -d`
3. Open web page: http://localhost:8080
4. Enter a question.


# Behavior
Doesn't always give the correct answer, and it's very, very slow on my computer.
I think it's partly due to the LLM, which probably requires more resources.


# NVidia CUDA
This file requires NVidia CUDA support to run
## Docker Desktop for Windows
1. Install your normal NVidia drivers for Windows
2. Follow these instructions: https://docs.nvidia.com/cuda/wsl-user-guide/index.html
> Note: I had to install GCC in WSL to get this to run:
> `sudo apt install gcc`
I also had issues with the latest (551) nvidia driver. Running `nvidia-smi` crashed in wsl, so I had to downgrade to version 537.


