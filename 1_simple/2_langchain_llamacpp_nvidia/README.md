# Overview
Example chatbot using LlamaCPP and LangChain using a model from HuggingFace.
We're using a 'quantized' model that will perform better on local hardware.
Configured to run CUDA instructions on an Nvidia GPU. You must have an Nvidia card to run this

# Running
1. Run `docker compose build`
2. Run `docker compose up -d`
3. Open web page: http://localhost:8080 and enter a question


# Behavior
Works pretty well. I don't notice a real difference between CPU mode, but I haven't tested extensively


# NVidia CUDA
This file requires NVidia CUDA support to run
## Docker Desktop for Windows
1. Install your normal NVidia drivers for Windows
2. Follow these instructions: https://docs.nvidia.com/cuda/wsl-user-guide/index.html
> Note: I had to install GCC in WSL to get this to run:
> `sudo apt install gcc`
I also had issues with the latest (551) nvidia driver. Running `nvidia-smi` crashed in wsl, so I had to downgrade to version 537.

# TODO
look into using LlamaCpp Grammers to constrain output (like SQL, Cypher, etc)