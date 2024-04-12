import measure
# langchain
from langchain.chains import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
# transformers
from ctransformers import AutoModelForCausalLM, AutoTokenizer
# huggingface
from huggingface_hub import snapshot_download, hf_hub_download
# web server
import gradio as gr

# Link to models: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
# Download the model
print("checking for model. Downloading if necessary...", flush=True)
repo_id = 'TheBloke/Llama-2-7B-Chat-GGUF'
model_file = 'llama-2-7b-chat.Q3_K_S.gguf'
modelpath = hf_hub_download(
    repo_id=repo_id,
    filename=model_file,
    cache_dir="/llm_cache"
    )
print('model path is:', model, flush=True)

# Create the model
print('Building LLM', flush=True)
model = AutoModelForCausalLM.from_pretrained(modelpath, model_type="llama", hf=True)
tokenizer = AutoTokenizer.from_pretrained(model)

def query(user_input, max_token_output=512):
    input_text = user_input
    input_ids = tokenizer(input_text, return_tensors='pt', truncation=False)
    input_ids["input_ids"] = input_ids["input_ids"]
    input_ids["attention_mask"] = input_ids["attention_mask"]
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