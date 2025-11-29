import torch
from transformers import AutoModelForCausalLM,
 AutoTokenizer, pipeline
from peft.peft_model import PeftModel

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

base_model_id = "CohereForAI/aya-expanse-8b"
our_model_id = "gaokerena/gaokerena-v1.0"
# our_model_id = "gaokerena/gaokerena-r1.0"

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=dtype,
    device_map=device
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

model = PeftModel.from_pretrained(model = model,
            model_id = our_model_id)
model = model.merge_and_unload()

pipe = pipeline("text-generation", model=model,
                tokenizer=tokenizer)
pipe_output = pipe([{"role": "user",
              "content": "your prompt!"}],
              max_new_tokens=1024,
              eos_token_id=[tokenizer.eos_token_id],
              do_sample=False,
)

output = pipe_output[0]["generated_text"][-1]["content"]
print(output)

