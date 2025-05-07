from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 选择GPT-2模型
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 设置 pad_token_id 为 eos_token_id 以外的标记
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# 编码输入文本
input_text = "Tell me a short bedtime story about a young man who is a member of the military."
inputs = tokenizer(input_text, return_tensors="pt")

# 生成文本
outputs = model.generate(
    inputs['input_ids'],
    max_length=500,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    pad_token_id=tokenizer.eos_token_id  # 设置 pad_token_id，通常为 eos_token_id
)

# 解码并输出生成的文本
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

