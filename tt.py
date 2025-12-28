from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the path you just saved from training
model_path = "./t5_result" 
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

text = "translate Chinese to English: 今天天气真不错。"
inputs = tokenizer(text, return_tensors="pt")

outputs = model.generate(**inputs, max_length=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))