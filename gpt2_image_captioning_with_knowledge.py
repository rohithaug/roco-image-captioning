import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
import json
import re

import evaluate
from PIL import Image
from configs import Configs
from transformers import VisionEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
os.environ["WANDB_DISABLED"] = "true"

import nltk
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=True)
    
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu

import warnings
warnings.filterwarnings("ignore")

# Hyperparameters

configs = Configs()

test_valid_percentage = 30 # (test - 15, valid - 15)

train_data_percentage = 100
valid_data_percentage = 100
test_data_percentage = 100

max_target_length = 256
random_state = 77

image_encoder_model = configs.image_encoder_model
text_decoder_model = configs.text_decoder_model

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(image_encoder_model, text_decoder_model)

# image feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(image_encoder_model)
# text tokenizer
tokenizer = AutoTokenizer.from_pretrained(text_decoder_model)

# GPT2 only has bos/eos tokens but not decoder_start/pad tokens
tokenizer.pad_token = tokenizer.eos_token

# update the model config
model.config.eos_token_id = tokenizer.eos_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

output_dir = "vit-gpt-model"
model.save_pretrained(output_dir)
feature_extractor.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

#####################################
#### Dataloading and Preparation ####
#####################################

image_dir = configs.train_image_dir
data_file = configs.train_image_data_file

data = pd.read_csv(data_file)

# Replace column name 'name' with 'image_path'
data["image_path"] = data.pop("name")

# Prepend 'image_dir' to all entries in 'image_path' column
data["image_path"] = image_dir + data["image_path"]

f = open(configs.llm_data_file, "r")
contents = f.read()
contents = contents.replace("\n", "")
json_data = json.loads(contents)

llm_df = pd.DataFrame(json_data)

llm_df = llm_df.drop('index', axis=1)

llm_df = llm_df[llm_df['relationship'].apply(lambda x: re.search(r'\w', str(x)) is not None)]
llm_df = llm_df.reset_index(drop=True)

data = data.merge(llm_df, on='id')

for index, row in data.iterrows():
    image_path = row['image_path']
    if not os.path.exists(image_path):
        data.drop(index, inplace=True)
    else:
        try:
            image = Image.open(image_path)
        except Exception:
            data.drop(index, inplace=True)
        
# Reset the index after dropping rows
data.reset_index(drop=True, inplace=True)

# Split data into train, test, and valid datasets
train_data, valid_test_data = train_test_split(data, test_size=test_valid_percentage/100, random_state=42)
valid_data, test_data = train_test_split(valid_test_data, test_size=0.5, random_state=42)

# Reset index
train_data = train_data.reset_index(drop=True)
valid_data = valid_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

print("Train data shape: ", train_data.shape)
print("Valid data shape: ", valid_data.shape)
print("Test data shape: ", test_data.shape)

# Select n% of data
train_data = train_data.sample(frac=train_data_percentage/100, random_state=42)
valid_data = valid_data.sample(frac=valid_data_percentage/100, random_state=42)
test_data = test_data.sample(frac=test_data_percentage/100, random_state=42)

from datasets import Dataset, DatasetDict

# Convert DataFrame to Hugging Face dataset dictionary format
train_data_dict = Dataset.from_pandas(train_data)
valid_data_dict = Dataset.from_pandas(valid_data)
test_data_dict = Dataset.from_pandas(test_data)

dataset_dict = DatasetDict({
    'train': train_data_dict,
    'validation': valid_data_dict,
    'test': test_data_dict
})

print(dataset_dict)

class ImageCaptioningDatasetWithKnowledge(Dataset):
    def __init__(self, ds, ds_type, max_target_length):
        self.ds = ds
        self.max_target_length = max_target_length
        self.ds_type = ds_type

    def __getitem__(self, idx):
        image_path = self.ds[self.ds_type]['image_path'][idx]
        caption = self.ds[self.ds_type]['caption'][idx]
        model_inputs = dict()
        model_inputs['labels'] = self.tokenization_fn(caption, self.max_target_length)
        model_inputs['pixel_values'] = self.feature_extraction_fn(image_path)
        return model_inputs

    def __len__(self):
        return len(self.ds[self.ds_type])
    
    # text preprocessing step
    def tokenization_fn(self, caption, max_target_length):
        """Run tokenization on caption."""
        labels = tokenizer(caption, 
                          padding="max_length", 
                          max_length=max_target_length,
                          truncation=True).input_ids

        return labels

    # image preprocessing step
    def feature_extraction_fn(self, image_path):
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        encoder_inputs = feature_extractor(images=image, return_tensors="np")

        return encoder_inputs.pixel_values[0]
    
train_ds = ImageCaptioningDatasetWithKnowledge(dataset_dict, 'train', max_target_length)
eval_ds = ImageCaptioningDatasetWithKnowledge(dataset_dict, 'validation', max_target_length)
test_ds = ImageCaptioningDatasetWithKnowledge(dataset_dict, 'test', max_target_length)

################################################
## Define seq2seq training argumentsPermalink ##
################################################

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    output_dir="./image-captioning-output",
)

metric = evaluate.load("rouge")

ignore_pad_token_for_loss = True

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds,
                                                     decoded_labels)

    result = metric.compute(predictions=decoded_preds,
                            references=decoded_labels,
                            use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    return result

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=feature_extractor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=default_data_collator,
)

print("Training the model...")

trainer.train()

trainer.save_model("./image-captioning-output")
tokenizer.save_pretrained("./image-captioning-output")

print("Testing the model on test data...")

# Get predictions from the model
predictions = trainer.predict(test_ds)

# Process and evaluate the predictions
preds = predictions.predictions
labels = predictions.label_ids

# Post-process the predictions and labels
decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

# Calculate evaluation metrics
bleu_scores = []
res = []

# Print the actual captions and predicted captions
i = 0
for actual_caption, predicted_caption in zip(decoded_labels, decoded_preds):
    i += 1
    bleu_score = sentence_bleu([actual_caption.split()], predicted_caption.split())
    bleu_scores.append(bleu_score)
    if i % 20 == 0:
        print("Actual Caption:", actual_caption)
        print("Predicted Caption:", predicted_caption)
        print("Blue score: ", bleu_score)
        print("--------------")

average_bleu_score = np.mean(bleu_scores)
print("Computed average BLEU score using GPT2 Image Captioning with LLM knowledge:", average_bleu_score)