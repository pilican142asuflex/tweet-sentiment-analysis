text = "I hate this world."

# Tokenize input
inputs = tokenizer(text, return_tensors="pt")

# Get model predictions
outputs = model(**inputs)
logits = outputs.logits

# Convert logits to probabilities (if needed)
import torch
probs = torch.nn.functional.softmax(logits, dim=-1)

print("Logits:", logits)
print("Probabilities:", probs)
