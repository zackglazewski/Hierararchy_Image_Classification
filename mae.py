from datasets import load_dataset, DatasetDict
from transformers import AutoImageProcessor, ViTForImageClassification, Trainer, TrainingArguments
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
from PIL import Image
import torch
import os


# Define preprocessing
print("Defining Processing Behavior")
image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean = image_processor.image_mean, std=image_processor.image_std)
])

def load_data(data_dir):
    dataset = load_dataset('imagefolder', data_dir)
    return dataset



# Load Data
print("Loading Data")

# Use absolute path for the dataset
data_dir = os.path.abspath("data_mini/2021_train_mini")
print(f"Loading dataset from: {data_dir}")

dataset = load_data(data_dir)
def transform_example(example):
    example['image'] = transform(example['image'])
    return example

dataset = dataset.with_transform(transform_example)


# Split dataset
print("Splitting Data")
train_test_split = dataset['train'].train_test_split(test_size=0.2)
dataset = DatasetDict({
    'train': train_test_split['train'],
    'validation': train_test_split['test']
})

labels = dataset['train'].features['label'].names




# Load Model
print("Loading Model")
model = ViTForImageClassification.from_pretrained("facebook/vit-mae-base", num_labels=len(labels))

training_args = TrainingArguments(
    output_dir = "./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = dataset['train'],
    eval_dataset = dataset['validation'],
    tokenizer = image_processor
)

print("Training")
trainer.train()