import warnings
warnings.filterwarnings('ignore')

from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Define the Hugging Face model name
# hf_model_name = "Qwen/Qwen2.5-3B-Instruct"  # Replace with a larger or more conversational model as needed
hf_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
model = AutoModelForCausalLM.from_pretrained(hf_model_name).to("cuda")  # Move model to GPU

# Create a pipeline for text generation with GPU support
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    truncation=True,
    device=0  # Use GPU (device index 0)
)

# Wrap the pipeline in a HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=hf_pipeline)