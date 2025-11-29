from transformers import AutoProcessor,
 AutoModelForImageTextToText
import torch
from peft.peft_model import PeftModel

base_model_id = "CohereForAI/aya-expanse-8b"
our_model_id = "gaokerena/gaokerena-v1.0"
#our_model_id = "gaokerena/gaokerena-r1.0"

processor = AutoProcessor.from_pretrained(base_model_id)
model = AutoModelForImageTextToText.from_pretrained(
    base_model_id, device_map="auto",
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(model=model,
                                  model_id=our_model_id)
model = model.merge_and_unload()

messages = [
    {"role": "user",
     "content": [
       {"type": "image", "url": "your image!"},
        {"type": "text", "text": "your prompt!"},
    ]},
    ]

inputs = processor.apply_chat_template(
    messages, padding=True, add_generation_prompt=True,
    tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device)

gen_tokens = model.generate(
    **inputs,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.3,
)
print(processor.tokenizer.decode(
    gen_tokens[0][inputs.input_ids.shape[1]:],
    skip_special_tokens=True))
