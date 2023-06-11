# Uncomment the following line to install the necessary packages
# !pip install accelerate sentencepiece evaluate absl-py nltk rouge_score

# Imports

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
import evaluate
from configs import Configs

# file paths for training and validation data

config = Configs()

base_path = config.base_path
train_input_file = config.train_input_file
train_target_file = config.train_target_file
valid_input_file = config.valid_input_file
valid_target_file = config.valid_target_file

train_input_data = [line.split() for line in open(train_input_file)]
train_input_data = [[t[0], " ".join(t[1:])] for t in train_input_data]
train_input_data = pd.DataFrame(train_input_data)
train_input_data.columns = ["id", "semtypes"]

train_target_data = pd.read_csv(train_target_file, sep="\t", header=None)
train_target_data.columns = ["id", "caption"]

valid_input_data = [line.split() for line in open(valid_input_file)]
valid_input_data = [[t[0], " ".join(t[1:])] for t in valid_input_data]
valid_input_data = pd.DataFrame(valid_input_data)
valid_input_data.columns = ["id", "semtypes"]

valid_target_data = pd.read_csv(valid_target_file, sep="\t", header=None)
valid_target_data.columns = ["id", "caption"]

# only the first 0.1% of the data for testing

portion = config.portion

train_input_data = train_input_data[: int(len(train_input_data) * portion)]
train_target_data = train_target_data[: int(len(train_target_data) * portion)]
valid_input_data = valid_input_data[: int(len(valid_input_data) * portion)]
valid_target_data = valid_target_data[: int(len(valid_target_data) * portion)]

# Creating custom Dataset class

class RoCoDataset(Dataset):
    def __init__(self, input_file, target_file, tokenizer):
        self.data = pd.merge(input_file, target_file, on="id", how="inner")
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        input_text = row["semtypes"]
        target_text = row["caption"]

        input_encoding = self.tokenizer.encode_plus(
            input_text, padding="max_length", max_length=128, truncation=True
        )
        target_encoding = self.tokenizer.encode_plus(
            target_text, padding="max_length", max_length=128, truncation=True
        )

        input_ids = input_encoding["input_ids"]
        input_attention_mask = input_encoding["attention_mask"]
        target_ids = target_encoding["input_ids"]
        target_attention_mask = target_encoding["attention_mask"]

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(input_attention_mask),
            "labels": torch.tensor(target_ids),
            "decoder_attention_mask": torch.tensor(target_attention_mask),
        }


tokenizer = T5Tokenizer.from_pretrained(config.pretrained_model)
model = T5ForConditionalGeneration.from_pretrained(config.pretrained_model)

device = config.device
model.to(device)

train_dataset = RoCoDataset(train_input_data, train_target_data, tokenizer)
valid_dataset = RoCoDataset(valid_input_data, valid_target_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=config.num_train_epochs,
    per_device_train_batch_size=config.batch_size,
    per_device_eval_batch_size=config.batch_size,
    save_steps=500,
    save_total_limit=2,
    overwrite_output_dir=True,
    learning_rate=config.learning_rate,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    disable_tqdm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the trained model
trainer.save_model("./trained_model")

# Test

print("Testing the model on test data...")

res = []

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")


def calculate_bleu_and_rouge(reference, hypothesis):
    bleu_score = bleu.compute(predictions=[hypothesis], references=[reference])
    rouge_score = rouge.compute(
        predictions=[hypothesis], references=[reference]
    )

    return {"bleu_score": bleu_score, "rouge_score": rouge_score}


# file paths for training and validation data

train_input_data = [line.split() for line in open(train_input_file)]
train_input_data = [[t[0], " ".join(t[1:])] for t in train_input_data]
train_input_data = pd.DataFrame(train_input_data)
train_input_data.columns = ["id", "semtypes"]

train_target_data = pd.read_csv(train_target_file, sep="\t", header=None)
train_target_data.columns = ["id", "caption"]

# last 10 rows of train_input_data
input_data = train_input_data.tail(50)
target_data = train_target_data.tail(50)

for _, row in input_data.iterrows():
    input_id = row["id"]
    input_text = row["semtypes"]

    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    input_ids = input_ids.to(device)
    model.to(device)

    # Generate captions using the model
    outputs = model.generate(input_ids)

    # Decode the generated captions
    generated_captions = tokenizer.decode(outputs[0], skip_special_tokens=True)

    target_caption = train_target_data[train_target_data["id"] == input_id][
        "caption"
    ].values[0]
    print("Input: ", input_text)
    print("Generated Caption: ", generated_captions)
    print("Original Caption: ", target_caption)

    # Calculating BLEU score and ROGUE score between the captions

    score = calculate_bleu_and_rouge(target_caption, generated_captions)
    bleu_score, rouge_score = score["bleu_score"], score["rouge_score"]

    temp = {
        "id": input_id,
        "input_text": input_text,
        "generated_caption": generated_captions,
        "target_caption": target_caption,
        "bleu_score": bleu_score,
        "rouge_score": rouge_score,
    }

    res.append(temp)


res = pd.DataFrame(res)
res.to_csv("./results.csv", index=False)
print(f"Results saved in ./results.csv")