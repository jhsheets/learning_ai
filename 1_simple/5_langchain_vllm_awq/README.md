# Overview
Example chatbot using langchain and transformers to load an AWQ quantized model in vLLM.

This builds vLLM with CUDA support, so you'll need an Nvidia card to run this.

# Running
1. Run `docker compose build`
2. Run `docker compose up -d`
3. Open web page: http://localhost:8080
4. Enter a question. There's an expandable section to show example queries


# Behavior
It takes a while for vLLM to load. I'm unsure if it's due to some settings I'm passing in, or if its related to the
warning that vLLM gives about *awq quantization is not fully optimized yet*, or if it's normal vLLM behavior, or maybe
it's just the behavior of trying to load these AWQ models.

Once it's loaded, it's giving good responses.

The responses using `TheBloke/Llama-2-7B-Chat-AWQ` seem to be nice and quick to generate, and appear fairly accurate.

My responses seem to get chopped-off if they're large. I increased the `max_new_tokens` value to 512, which
helped. I'm not sure how much higher it can go. It's probably part of the model spec. This does increase the amount of 
time it takes to generate the response, but at least the response doesn't appear truncated.

# Notes

* vLLM Docs: https://docs.vllm.ai/en/latest/models/engine_args.html
* vLLM with LangChain: https://python.langchain.com/docs/integrations/llms/vllm/

## Llama model
https://huggingface.co/TheBloke/Llama-2-7B-Chat-AWQ

This is the model that this app is using.

First I tried this template:
```
template = """Question: {question}

Answer: """
```
It worked, but I changed it to the one I've checked in based on the model's documentation on huggingface.co.


## Mistral model
https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-AWQ

I also tried using the `TheBloke/Mistral-7B-Instruct-v0.1-AWQ` model with the exact same code. 
The model still took a long time to load, but it was significantly faster than `TheBloke/Llama-2-7B-Chat-AWQ` to load.
However, the responses took a bit longer to generate.

First I tried it with this prompt:
```
template = """Question: {question}

Answer: """
```
It gave very weird answers.

I ended up changing it to this:
> template=""""<|im_start|>system
> You will try to answer the question to the best of your ability<|im_end|>
> <|im_start|>user
> {question}<|im_end|>
> <|im_start|>assistant
> <|im_end|>
> """

This prompt seemed to work well, but it did include the original question in the response. 

The docs for this model on huggingface.co state that the model should actually look like the one I'm using with the 
`TheBloke/Llama-2-7B-Chat-AWQ` model, but I didn't try it.