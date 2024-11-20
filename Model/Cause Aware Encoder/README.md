# Cause Aware Encoder
---

The provided code contains a series of functionalities for processing data with pretrained language models like COMET and BlenderBot, as part of an emotional support analysis system. Here’s an explanation and cleanup for better clarity:

---

### Key Functionalities:

#### 1. **COMET Module**
   - Downloads and initializes the COMET model (`comet-atomic_2020_BART`).
   - Defines intra- and inter-relational concepts (`xReact`, `oWant`, etc.) for generating causal and contextual knowledge from text data.
   - Processes seeker and supporter utterances in a dataset (`ESConv.json`) to:
     - Generate text-based inferences for defined relations.
     - Extract hidden states from the encoder for deeper analysis.

#### 2. **Encoder Module**
   - Utilizes BlenderBot (`blenderbot_small-90M`) for analyzing emotional support conversations.
   - Extracts embeddings for "situations" and "contexts" from the dataset, enabling downstream tasks like similarity analysis or clustering.

#### 3. **Utilities**
   - **Batch Processing:** Efficiently handles large datasets by splitting them into manageable chunks.
   - **Hidden State Extraction:** Retrieves encoder outputs, useful for representing input data in high-dimensional space.
   - **Preprocessing:** Combines dialogues into single contexts for modeling.

---

#### **COMET Processing Module**
```python
import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

class CometProcessor:
    def __init__(self, model_path, batch_size=1):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.batch_size = batch_size

    def generate(self, queries, num_generate=1):
        results = []
        with torch.no_grad():
            for batch in tqdm(self._chunks(queries, self.batch_size), desc="Generating"):
                tokenized = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="longest").to(self.device)
                generated = self.model.generate(
                    input_ids=tokenized['input_ids'],
                    attention_mask=tokenized['attention_mask'],
                    num_beams=num_generate,
                    num_return_sequences=num_generate
                )
                results.extend(self.tokenizer.batch_decode(generated, skip_special_tokens=True))
        return results

    def get_hidden_states(self, queries):
        with torch.no_grad():
            tokenized = self.tokenizer(queries, return_tensors="pt", truncation=True, padding="longest").to(self.device)
            outputs = self.model.model.encoder(input_ids=tokenized['input_ids'], attention_mask=tokenized['attention_mask'])
            return outputs.last_hidden_state

    @staticmethod
    def _chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

def process_comet_data(file_path, model_path, relations):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    comet = CometProcessor(model_path)
    processed_results = []

    for conv in tqdm(data, desc="Processing Conversations"):
        seeker_utterances = [utt['content'] for utt in conv['dialog'] if utt['speaker'] == 'seeker']
        conversation_results = []

        for utterance in seeker_utterances:
            utterance_results = {}
            for rel in relations:
                query = f"{utterance} [MASK] {rel}"
                generated = comet.generate([query])
                hidden_states = comet.get_hidden_states([query])
                utterance_results[rel] = {"generated": generated, "hidden_states": hidden_states.shape}
            conversation_results.append(utterance_results)

        processed_results.append(conversation_results)
    return processed_results
```

---

#### **BlenderBot Encoder Module**
```python
import json
import torch
from transformers import BlenderbotTokenizer, BlenderbotModel
from tqdm import tqdm

class EmotionalSupportEncoder:
    def __init__(self, model_name, max_length=128, batch_size=8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
        self.model = BlenderbotModel.from_pretrained(model_name).to(self.device)
        self.max_length = max_length
        self.batch_size = batch_size

    def encode_texts(self, texts):
        embeddings = []
        for batch in tqdm(self._batchify(texts, self.batch_size), desc="Encoding Texts"):
            tokenized = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**tokenized)
                embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu())
        return torch.cat(embeddings, dim=0)

    @staticmethod
    def _batchify(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def process_dataset(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        situations = [entry['situation'] for entry in data]
        contexts = [" ".join([utt['content'] for utt in entry['dialog']]) for entry in data]

        situation_embeddings = self.encode_texts(situations)
        context_embeddings = self.encode_texts(contexts)

        return situation_embeddings, context_embeddings
```

---
