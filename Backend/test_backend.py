

import json

def load_json_file(file_path):
  """Loads a JSON file from the specified path.

  Args:
    file_path: The path to the JSON file.

  Returns:
    A Python object representing the JSON data.
  """
  with open(file_path, 'r') as f:
    data = json.load(f)
  return data

# Example usage:
# Assuming you have a file named 'data.json' in your current directory.
data = load_json_file('/content/extracted_data.json')
print(data)

sit_experience = []
for i in data:
# Extracting the required fields
    extracted_data = {
        "response": i.get("response"),
        "question": i.get("post")
    }

    # Output the extracted data
    sit_experience.append(extracted_data)

sit_experience



from transformers import AutoModelForCausalLM, AutoTokenizer

import pandas as pd
from transformers import T5Tokenizer

# Sample dataset
data = sit_experience

# Convert list of dicts to a pandas DataFrame
df = pd.DataFrame(data)

# Initialize the T5 tokenizer (you can also use BART tokenizer if preferred)
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Define a function to preprocess and tokenize the dataset
def preprocess_and_tokenize(df):
    # Tokenizing function to apply to each row
    def tokenize_row(row):
        # Tokenize the input and output sequences
        input_text = row['question']
        output_text = row['response']  # Assuming summary is same as situation for this example

        # Tokenize the input and output sequences
        inputs = tokenizer(input_text, truncation=True, padding='max_length', max_length=512)
        outputs = tokenizer(output_text, truncation=True, padding='max_length', max_length=50)

        # Add 'labels' for training (output labels are the tokenized summary)
        inputs['labels'] = outputs['input_ids']
        return inputs

    # Apply the tokenization to the entire DataFrame
    tokenized_data = df.apply(tokenize_row, axis=1)

    # Convert the list of tokenized data into a DataFrame
    tokenized_df = pd.DataFrame(tokenized_data.tolist())
    return tokenized_df

# Apply preprocessing and tokenization
tokenized_df = preprocess_and_tokenize(df)

# Show the tokenized DataFrame
print(tokenized_df.head())

!pip install datasets

from datasets import Dataset

# Assume 'situation' is the long text and we create a 'summary' (for now, we use the 'situation' as is)
train_data = [{"long_sentence": row['question'], "short_summary": row['response']} for index, row in df.iterrows()]

# Convert the list of dictionaries to a Dataset object
train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))

# Tokenization function
def preprocess_function(examples):
    inputs = tokenizer(examples['long_sentence'], truncation=True, padding='max_length', max_length=512)
    labels = tokenizer(examples['short_summary'], truncation=True, padding='max_length', max_length=50)
    inputs['labels'] = labels['input_ids']
    return inputs

# Apply the function to the dataset
train_dataset = train_dataset.map(preprocess_function, batched=True)

from sklearn.model_selection import train_test_split

# Split the DataFrame into train and eval sets (80% train, 20% eval)
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

# Show the first few rows of both sets
print("Training Set:")
print(train_df.head())
print("\nEvaluation Set:")
print(eval_df.head())

# Tokenize the train and eval DataFrames
train_tokenized_df = preprocess_and_tokenize(train_df)
eval_tokenized_df = preprocess_and_tokenize(eval_df)

# Show the tokenized data
print("Tokenized Training Set:")
print(train_tokenized_df.head())
print("\nTokenized Evaluation Set:")
print(eval_tokenized_df.head())

from datasets import Dataset

# Convert the tokenized DataFrames to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_tokenized_df)
eval_dataset = Dataset.from_pandas(eval_tokenized_df)

# Show the first few rows of the train dataset
print("Train Dataset:", train_dataset)

import os
os.environ["WANDB_DISABLED"] = "true"

import torch

from transformers import Trainer, TrainingArguments

# Example model
from transformers import T5ForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("t5-small")
model.to(device)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="./results",  # output directory
    evaluation_strategy="epoch",  # evaluate every epoch
    per_device_train_batch_size=8,  # batch size for training
    per_device_eval_batch_size=8,   # batch size for evaluation
    num_train_epochs=20,  # number of training epochs
    logging_dir="./logs",  # directory for storing logs
    logging_steps=10,  # log every 10 steps
    report_to=None,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Pass eval dataset here
)

# Start training
trainer.train()

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Check if GPU is available and select the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
# model = T5ForConditionalGeneration.from_pretrained("t5-small")
model.to(device)  # Move model to the selected device (GPU or CPU)
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Summarization function
def summarize(text):
    # Preprocess the input text
    inputs = tokenizer(text , return_tensors="pt", max_length=512, truncation=True)

    # Move the input tensor to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate the summary
    summary_ids = model.generate(inputs["input_ids"], max_length=50, min_length=10, length_penalty=2.0, num_beams=4)

    # Decode and return the summary
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Example usage
test_sentence = "i am going to lucky i think"
print(summarize(test_sentence))

# prompt: save the model so that i can share it people

# Save the model and tokenizer
model.save_pretrained("./my_summarization_model")
tokenizer.save_pretrained("./my_summarization_model")

# Optionally, you can also save the entire training state
# trainer.save_model("./my_summarization_model")

print("Model saved successfully!")

# To share the model:
# 1. Zip the saved model directory.
# 2. Upload the zipped model to a cloud storage service (e.g., Google Drive).
# 3. Share the link with others.

# prompt: how they can run for inference

# Load the saved model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("./my_summarization_model")
tokenizer = T5Tokenizer.from_pretrained("./my_summarization_model")

# Check if GPU is available and select the device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)  # Move model to the selected device (GPU or CPU)


# Summarization function (same as before)
def summarize(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    summary_ids = model.generate(inputs["input_ids"], max_length=50, min_length=10, length_penalty=2.0, num_beams=4)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# Example usage with a new sentence
new_sentence = "I had a really bad day at work. I got yelled at by my boss and I spilled coffee all over my computer."
summary = summarize(new_sentence)
summary
