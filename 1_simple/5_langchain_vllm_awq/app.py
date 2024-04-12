import measure
import time
# langchain
from langchain.chains import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain_community.llms import VLLM
# vllm
from vllm import LLM, SamplingParams
# transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# huggingface
from huggingface_hub import snapshot_download, hf_hub_download
# web server
import gradio as gr


# Download the model
print("checking for model. Downloading if necessary...", flush=True)
repo_id = 'TheBloke/Llama-2-7B-Chat-AWQ'
modelpath = snapshot_download(
    repo_id=repo_id,
    cache_dir="/llm_cache"
    )
print('model path is:', modelpath, flush=True)



# Create the model
print('Loading LLM. This may take a while...', flush=True)
model_start = time.time()
llm = VLLM(
    model=modelpath,
    trust_remote_code=True,  # mandatory for hf models
    max_new_tokens=512,
    dtype='half',
    top_k=10,
    top_p=0.95,
    temperature=0.8,
    vllm_kwargs={
        "max_model_len": 1000, # You may need to adjust this to fit on your pc; you'll get an error explaining if you do
        "enforce_eager": True, # Disabling this incurs a long load-up time, but will use CUDA graph optimizations and give potentially better performance but uses more GPU memory
    },
)
model_end = time.time()
print('Model loaded in {:.2f} seconds'.format(model_end - model_start), flush=True)


template = """[INST] <<SYS>>
You are a helpful and honest assistant.
<</SYS>>
{question}[/INST]"""


prompt = PromptTemplate(template=template, input_variables=["question"])


# Set callback function for gradio
def chatbot(user_input, history):
    measurement = measure.measure_start()
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    answer = llm_chain.invoke(user_input)
    measure.measure_end(*measurement)
    # answer contains 'question' and 'text' keys. We only want the text.
    return answer['text']

# Setup gradio
demo_chatbot = gr.ChatInterface(chatbot, title="Chatbot", description="Enter text to start chatting.")

# Start gradio
demo_chatbot.launch(server_name="0.0.0.0", server_port=8080, enable_queue=True, share=False)


