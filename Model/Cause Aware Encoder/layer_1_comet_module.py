# -*- coding: utf-8 -*-
"""Layer 1. COMET Module.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1vDl6NG4-xOqVFs8nDTmPT8KwbehFkoPx
"""

!wget https://storage.googleapis.com/ai2-mosaic-public/projects/mosaic-kgs/comet-atomic_2020_BART.zip

!unzip comet-atomic_2020_BART.zip -d comet-atomic_2020_BART

import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

def use_task_specific_params(model, task):
    task_specific_params = {
        "summarization": {
            "max_length": 512,
            "min_length": 50,
            "num_beams": 5,
            "length_penalty": 2.0,
            "early_stopping": True,
        },
    }
    model.config.update(task_specific_params[task])

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]

class Comet:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        task = "summarization"
        use_task_specific_params(self.model, task)
        self.batch_size = 1
        self.decoder_start_token_id = None

    def generate(self, queries, decode_method="beam", num_generate=5):
        with torch.no_grad():
            examples = queries
            decs = []
            for batch in list(chunks(examples, self.batch_size)):
                batch = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(self.device)
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']

                # Generate outputs from the model
                summaries = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=self.decoder_start_token_id,
                    num_beams=num_generate,
                    num_return_sequences=num_generate,
                )

                # Decode the output
                dec = self.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                decs.append(dec)
            return decs

    def get_encoder_hidden_states(self, queries):
        """Extract hidden states from the encoder's last layer"""
        with torch.no_grad():
            batch = self.tokenizer(queries, return_tensors="pt", truncation=True, padding="max_length").to(self.device)
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            # Extract the hidden states from the encoder
            # Access the encoder through self.model.model.encoder for BartForConditionalGeneration
            outputs = self.model.model.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            hidden_states = outputs.last_hidden_state
            return hidden_states

# Define relations
intra_relations = ["xReact", "xEffect", "xWant"]
inter_relations = ["oReact", "oEffect", "oWant"]

def run():
    # Load dataset (e.g., esconv_data)
    with open('/content/ESConv.json', 'r',encoding='utf-8') as f:
        esconv_data = json.load(f)

    print("model loading ...")
    comet = Comet("/content/comet-atomic_2020_BART/comet-atomic_2020_BART")
    comet.model.zero_grad()
    print("model loaded")
    K_EC_intra = []
    # Iterate over the data and generate COMET inputs
    for conversation in tqdm(esconv_data[:5]):
        situation = conversation['situation']
        seeker_utterances = []
        supporter_utterances = []
        for utt in conversation['dialog']:
            if utt['speaker']=='seeker':
                seeker_utterances.append(utt['content'])
            else:
                supporter_utterances.append(utt['content'])
        all_hidden = []
        # Loop through seeker's utterances
        for utterance in seeker_utterances:
            for rel in intra_relations:
                # Create the query: Concatenate utterance with relation
                query = f"{utterance} [MASK] {rel}"
                print(f"Query: {query}")

                # Generate descriptions using COMET for the current relation
                results = comet.generate([query], decode_method="beam", num_generate=1)
                print(f"Results for {rel}: {results}")

                # Get hidden states for causal representations
                hidden_states = comet.get_encoder_hidden_states([query])
                print(f"Hidden States for {rel}: {hidden_states.shape}")
                all_hidden.append(hidden_states)
        K_EC_intra.append(hidden_states)
    #print(K_EC_intra)
    return K_EC_intra
if __name__ == "__main__":
    K_EC_intra = run()
    print(K_EC_intra)

import json
from tqdm import tqdm

# Import Comet from the previous cell or file where it is defined
# from your_file import Comet  # Assuming your_file contains the Comet class

def run_st_inter():
    # Load dataset (e.g., esconv_data)
    with open('/content/ESConv.json', 'r', encoding='utf-8') as f:
        esconv_data = json.load(f)

    print("Model loading...")
    # Change the model path to the correct location
    comet = Comet("/content/comet-atomic_2020_BART")  # This was the path cloned in input 18
    comet.model.zero_grad()
    print("Model loaded")

    K_EC_inter = []
    K_ES = []


    # Iterate over the data and generate COMET inputs
    for conversation in tqdm(esconv_data[:5]):
        situation = conversation['situation']
        all_situation_hidden = []

        # Generate COMET queries for the situation using each relation
        for rel in inter_relations:
            # Create the query: Concatenate situation with relation
            situation_query = f"{situation} [MASK] {rel}"
            print(f"Situation Query: {situation_query}")

            # Generate descriptions using COMET for the current relation
            situation_results = comet.generate([situation_query], decode_method="beam", num_generate=1)
            print(f"Situation Results for {rel}: {situation_results}")

            # Get hidden states for the situation
            situation_hidden_states = comet.get_encoder_hidden_states([situation_query])
            print(f"Situation Hidden States for {rel}: {situation_hidden_states.shape}")

            all_situation_hidden.append(situation_hidden_states)

        # Process dialog
        seeker_utterances = []
        supporter_utterances = []
        for utt in conversation['dialog']:
            if utt['speaker'] == 'seeker':
                seeker_utterances.append(utt['content'])
            else:
                supporter_utterances.append(utt['content'])

        all_hidden = []

        # Loop through supporter's utterances
        for utterance in supporter_utterances:
            utterance_hidden_states = []

            # Generate COMET queries for each relation
            for rel in inter_relations:
                # Create the query: Concatenate utterance with relation
                query = f"{utterance} [MASK] {rel}"
                print(f"Query: {query}")

                # Generate descriptions using COMET for the current relation
                results = comet.generate([query], decode_method="beam", num_generate=1)
                print(f"Results for {rel}: {results}")

                # Get hidden states for causal representations
                hidden_states = comet.get_encoder_hidden_states([query])
                print(f"Hidden States for {rel}: {hidden_states.shape}")

                utterance_hidden_states.append(hidden_states)

            # Append the hidden states for all relations for the current utterance
            all_hidden.append(utterance_hidden_states)

        # Append the hidden states for each conversation
        K_EC_inter.append(all_hidden)
        K_ES.append(all_situation_hidden)

    print("K_EC_inter:", K_EC_inter)
    print("K_ES:", K_ES)
    return K_EC_inter, K_ES

if __name__ == "__main__":
    K_EC_inter = run()
    print(K_EC_intra)
    K_ES = run()
    print(K_ES)

