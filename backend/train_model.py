import pandas as pd
import torch
import os
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("dataset/tweets.csv")

# Ensure necessary columns exist
if "statement" not in df.columns or "status" not in df.columns:
    raise KeyError("Dataset must contain 'statement' and 'status' columns.")

# Drop missing values
df = df.dropna(subset=["statement", "status"])

# Encode labels
label_mapping = {
    "Normal": 0, "Depression": 1, "Suicidal": 2, "Anxiety": 3, 
    "Stress": 4, "Bi-Polar": 5, "Personality Disorder": 6
}
df["label"] = df["status"].map(label_mapping).fillna(0).astype(int)  # Ensure no NaN values

# Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["statement"].astype(str).tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

# Tokenization
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# Convert to PyTorch dataset
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, num_labels=7):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

        # Convert labels to one-hot encoding for BCE loss
        self.one_hot_labels = F.one_hot(self.labels, num_classes=num_labels).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx], dtype=torch.long),
            "labels": self.one_hot_labels[idx],  # One-hot labels for BCE loss
        }

train_dataset = TweetDataset(train_encodings, train_labels)
val_dataset = TweetDataset(val_encodings, val_labels)

# Load pre-trained model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=7)

# Define training parameters
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_total_limit=2,
    logging_dir="./logs"
)

# Custom Trainer to use BCE Loss
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False,**kwargs):
        labels = inputs.pop("labels")  # Extract labels
        outputs = model(**inputs)  # Get model outputs
        logits = outputs.logits  # Extract logits

        # Use BCE Loss
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        
        return (loss, outputs) if return_outputs else loss

# Train model
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

# Save model
os.makedirs("models/bert_sentiment", exist_ok=True)
model.save_pretrained("models/bert_sentiment")
tokenizer.save_pretrained("models/bert_sentiment")

print("Model training completed!")
