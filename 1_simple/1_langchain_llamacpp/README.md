# Overview
Example chatbot using LlamaCPP and LangChain using a model from HuggingFace.
We're using a 'quantized' model that will perform better on local hardware.


# Running
1. Run `docker compose build`
2. Run `docker compose up -d`
3. Open web page: http://localhost:8080 and enter a question


# Behavior
Works pretty well. I don't notice a real difference between CPU mode, but I haven't tested extensively