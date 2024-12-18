# -*- coding: utf-8 -*-
"""Layer 1. Encoder Module.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1vDl6NG4-xOqVFs8nDTmPT8KwbehFkoPx
"""

import json
import torch
from transformers import BlenderbotTokenizer, BlenderbotModel
from typing import List, Tuple, Dict
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionalSupportAnalyzer:
    def __init__(
        self,
        model_name: str = '/content/blenderbot_small-90M',
        max_length: int = 128,
        batch_size: int = 8
    ):
        """
        Initialize the analyzer with memory-efficient settings.

        Args:
            model_name: Name of the BlenderBot model
            max_length: Maximum sequence length
            batch_size: Size of batches for processing
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.max_length = max_length

        logger.info(f"Using device: {self.device}")
        logger.info(f"Batch size: {batch_size}")

        try:
            self.tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
            self.model = BlenderbotModel.from_pretrained(model_name).to(self.device)
        except Exception as e:
            logger.error(f"Error loading BlenderBot model: {str(e)}")
            raise

    def load_dataset(self, file_path: str) -> Dict:
        """Load the dataset from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded dataset with {len(data)} entries")
            return data
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    def preprocess_data(self, data: List[Dict]) -> Tuple[List[str], List[str]]:
        """Extract situations and contexts from the dataset."""
        situations = []
        contexts = []

        for entry in data:
            situation = entry.get('situation', '')
            dialog = entry.get('dialog', [])

            context = " ".join([
                d.get('content', '') for d in dialog
                if isinstance(d, dict) and 'content' in d
            ])

            situations.append(situation)
            contexts.append(context)

        return situations, contexts

    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """Encode a single batch of texts."""
        encoded_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encoded_inputs)
            # Get the last hidden state for each sequence (mean pooling)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu()

        # Clear CUDA cache
        torch.cuda.empty_cache()
        return embeddings

    def encode_texts_batched(self, texts: List[str]) -> torch.Tensor:
        """Encode texts in batches to manage memory."""
        all_embeddings = []

        for i in tqdm(range(0, len(texts), self.batch_size), desc="Processing batches"):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.encode_batch(batch_texts)
            all_embeddings.append(batch_embeddings)

        return torch.cat(all_embeddings, dim=0)

    def process_conversations(self, file_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process all conversations with batch processing."""
        # Load and preprocess data
        data = self.load_dataset(file_path)
        situations, contexts = self.preprocess_data(data)

        logger.info("Generating situation embeddings...")
        situation_embeddings = self.encode_texts_batched(situations)

        logger.info("Generating context embeddings...")
        context_embeddings = self.encode_texts_batched(contexts)

        logger.info(f"Final shapes - Situations: {situation_embeddings.shape}, Contexts: {context_embeddings.shape}")
        return situation_embeddings, context_embeddings

def save_embeddings(embeddings: torch.Tensor, file_path: str):
    """Save embeddings to disk."""
    torch.save(embeddings, file_path)
    logger.info(f"Saved embeddings to {file_path}")

def load_embeddings(file_path: str) -> torch.Tensor:
    """Load embeddings from disk."""
    embeddings = torch.load(file_path)
    logger.info(f"Loaded embeddings of shape {embeddings.shape}")
    return embeddings

# Example usage
if __name__ == "__main__":
    # Initialize with smaller batch size
    analyzer = EmotionalSupportAnalyzer(batch_size=8)
    file_path = '/content/ESConv.json'

    try:
        # Process conversations
        situation_embeddings, context_embeddings = analyzer.process_conversations(file_path)

        # Save embeddings
        save_embeddings(situation_embeddings, 'situation_embeddings.pt')
        save_embeddings(context_embeddings, 'context_embeddings.pt')

        logger.info("Processing completed successfully")

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")