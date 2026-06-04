from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch


cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")
if cuda_available:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
else:
    print("Running on CPU (no NVIDIA GPU detected)")

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto" if cuda_available else "cpu",
    torch_dtype=torch.float16 if cuda_available else torch.float32,
)

prompt = "Explain binary search in simple terms."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=100
)

print(
    tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
)