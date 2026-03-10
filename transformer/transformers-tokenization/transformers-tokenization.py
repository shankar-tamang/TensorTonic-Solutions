import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        special_tokens = [self.pad_token, 
                         self.unk_token, 
                         self.bos_token, 
                         self.eos_token]

        for idx, token in enumerate(special_tokens):
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token

        current_id = len(special_tokens)

        vocab = set()

        for text in texts:
            words = text.split()
            vocab.update(words)

        for word in vocab:
            if word not in self.word_to_id:
                self.word_to_id[word] = current_id
                self.id_to_word[current_id] = word
                current_id += 1

        self.vocab_size = current_id

        
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        words = text.split()
        ids = []

        # for word in words:
        #     if word in self.word_to_id
        #         ids.append(self.word_to_id[word])
        #     else:
        #         ids.append(self.word_to_id[self.unk_token])

        ids = [self.word_to_id[word] if word in self.word_to_id else self.word_to_id[self.unk_token] for word in words]

        return ids
            
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        text = [self.id_to_word.get(id, self.unk_token) for id in ids]

        text = " ".join(text)
        return text
