# Overview
Example chatbot using langchain and transformers to load a quantized model.
The idea was to compare:
1. How well this (TheBloke/Llama-2-7B-Chat-GGUF) LLM compared to (TencentARC/LLaMA-Pro-8B-Instruct) in `3_langchain_huggingface_nvidia`
2. How well the same LLM performed when loaded through huggingface vs llamacpp


# Running
1. Run `docker compose build`
2. Run `docker compose up -d`
3. Open web page: http://localhost:8080
4. Enter a question. There's an expandable section to show example queries


# Behavior
First, this is much faster than `3_langchain_huggingface_nvidia`, but not as fast as `1_langchain_llamacpp`

Second, this is extremely verbose to any question, and sometimes answers strangely to even simple things like 'hi'.
Loves to answer in German.
Maybe this is because I wasn't passing in any prompts?
Even though I `had max_token_output=1024` at one point it threw errors if the response was more than 512 characters.

Also, I could not get this to work with Nvidia CUDa, even when my code was as close to identical as I could get when
compared to `3_langchain_huggingface_nvidia`.
It's using a `ctransformers` instead of `transformers` so that it can work with the GGUF model type.
However, I just get this error:
```
Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```



