# Code copied from:
# https://huggingface.co/spaces/TencentARC/LLaMA-Pro-8B-Instruct-Chat/blob/main/app.py
import measure
# langchain
from langchain.chains import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
# transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
# pytorch
import torch
# misc
import os
# web server
import gradio as gr

MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

# TODO: use hf_hub_download to download
model_id = 'TencentARC/LLaMA-Pro-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="/llm_cache", use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir="/llm_cache")

model.half().cuda()


@torch.inference_mode()
def query(user_input, max_token_output=1024):
    input_text = user_input
    input_ids = tokenizer(input_text, return_tensors='pt', truncation=False)
    input_ids["input_ids"] = input_ids["input_ids"].cuda()
    input_ids["attention_mask"] = input_ids["attention_mask"].cuda()
    generatedIds = model.generate(input_ids.input_ids, max_new_tokens=max_token_output, do_sample=False)
    result = tokenizer.batch_decode(generatedIds, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return result


# Set callback function for gradio
def chatbot(user_input, history):
    measurement = measure.measure_start()
    answer = query(user_input)
    measure.measure_end(*measurement)
    return answer

# Setup gradio
demo_chatbot = gr.ChatInterface(chatbot, title="Chatbot", description="Enter text to start chatting.")

# Start gradio
demo_chatbot.launch(server_name="0.0.0.0", server_port=8080, enable_queue=True, share=False)

