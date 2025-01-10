from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# model_llm_name = "microsoft/Phi-3.5-mini-instruct"
# model_llm_name = "HuggingFaceTB/SmolLM-1.7B"
model_llm_name = "Qwen/Qwen2.5-1.5B-Instruct"

model_llm = AutoModelForCausalLM.from_pretrained(
    model_llm_name, 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
tokenizer_llm = AutoTokenizer.from_pretrained(model_llm_name)



# messages = [
#     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#     {"role": "user", "content": "Who are you?"},
# ]

# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
# )

# generation_args = {
#     "max_new_tokens": 500,
#     "return_full_text": False,
#     "temperature": 0.0,
#     "do_sample": False,
# }

# output = pipe(messages, **generation_args)
# print(output[0]['generated_text'])
