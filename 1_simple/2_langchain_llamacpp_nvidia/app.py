import measure
# langchain imports
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
# for llm download
from huggingface_hub import snapshot_download, hf_hub_download
# web server
import gradio as gr


# Link to models: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
# Download the model
print("checking for model. Downloading if necessary...", flush=True)
# I can use snapshot_download to download an entire file, but I need to pass a specific gguf file to LlamaCpp's model_path
model = hf_hub_download(
    repo_id="TheBloke/Llama-2-7B-Chat-GGUF", 
    filename='llama-2-7b-chat.Q3_K_S.gguf', 
    cache_dir="/llm_cache/"
    )
print('model path is:', model, flush=True)


# Configure prompt
# TODO: I don't think this is a good prompt for the llm....
template = """Question: {question}

Answer: """

prompt = PromptTemplate(template=template, input_variables=["question"])


# https://api.python.langchain.com/en/latest/llms/langchain_community.llms.llamacpp.LlamaCpp.html
print('Building LLM', flush=True)
n_gpu_layers = 60       # Change this value based on your model and your GPU VRAM pool. Layers defined go on GPU, the rest on CPU. -1 puts all layers in GPU
n_batch = 100           # Should be between 1 and n_ctx (who's default is 512), consider the amount of VRAM in your GPU.
llm = LlamaCpp(
    model_path=model,
    n_gpu_layers=n_gpu_layers,          # comment out this line if using CPU inference instead of GPU
    n_batch=n_batch,                    # comment out this line if using CPU inference instead of GPU
    verbose=True,
    max_tokens=2000,
    # temperature=1
)

# Set callback function for gradio
def chatbot(question, history):
    measurement = measure.measure_start()
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    answer = llm_chain.run(question)
    measure.measure_end(*measurement)
    return answer


# Setup gradio
demo_chatbot = gr.ChatInterface(chatbot, title="Chatbot", description="Enter text to start chatting.")

# Start gradio
demo_chatbot.launch(server_name="0.0.0.0", server_port=8080, enable_queue=True, share=False)