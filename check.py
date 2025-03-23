from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Path to your model
model_path = "backend/models/bert_sentiment"

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Model and tokenizer loaded successfully!")

text="when i’m hurt, i shut down, i turn into a total bitch i shut off my emotions i act differently towards everything and everyone and i hate it."

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
# Get predicted class index
predicted_class = torch.argmax(probs, dim=1).item()
label_map = ["Normal", "Depression", "Suicidal", "Anxiety", "Stress", "Bi-Polar", "Personality Disorder"]
# Map index to label
predicted_label = label_map[predicted_class]

print("Predicted Mental Health Condition:", predicted_label)

