from transformers import AutoTokenizer, CLIPTextModel

model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
print(inputs)
outputs = model(output_hidden_state = 512, **inputs)
last_hidden_state = outputs.last_hidden_state
pooled_output = outputs.pooler_output

print(pooled_output.shape)