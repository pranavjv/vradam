import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
from collections import Counter, OrderedDict
import nltk
import random
from nltk.tokenize import word_tokenize

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class Vocab:
    """Simple vocabulary class as a replacement for torchtext.vocab"""
    def __init__(self, counter, max_size=None, min_freq=1, specials=('<unk>', '<pad>')):
        self.specials = specials
        self.itos = list(specials)
        self.stoi = {token: i for i, token in enumerate(self.itos)}
        
        # Add tokens from counter
        for token, count in counter.most_common(max_size):
            if count >= min_freq and token not in self.stoi:
                self.itos.append(token)
                self.stoi[token] = len(self.itos) - 1
        
        self.unk_index = 0 if '<unk>' in specials else None
        
    def __len__(self):
        return len(self.itos)
    
    def __getitem__(self, token):
        return self.stoi.get(token, self.unk_index)

class IMDBDataset(Dataset):
    """IMDB dataset implementation using HuggingFace datasets"""
    def __init__(self, split='train', max_vocab_size=25000):
        # Load from Hugging Face datasets
        if split == 'train' or split == 'val':
            # For validation, we'll split the training data later
            self.dataset = load_dataset('imdb', split='train')
        else:
            self.dataset = load_dataset('imdb', split='test')
        
        # Process text and labels
        self.texts = []
        self.labels = []
        
        for item in self.dataset:
            self.texts.append(item['text'])
            self.labels.append(1 if item['label'] == 1 else 0)  # 1 for positive, 0 for negative
        
        # Create vocabulary if it's training data
        if split == 'train' or split == 'val':
            # Split into train/val if needed
            if split == 'val':
                train_size = int(0.8 * len(self.texts))
                self.texts = self.texts[train_size:]
                self.labels = self.labels[train_size:]
            elif split == 'train':
                train_size = int(0.8 * len(self.texts))
                self.texts = self.texts[:train_size]
                self.labels = self.labels[:train_size]
                
            # Build vocabulary from training text
            counter = Counter()
            for text in self.texts:
                tokens = word_tokenize(text.lower())
                counter.update(tokens)
            
            # Create vocabulary
            self.vocab = Vocab(counter, max_size=max_vocab_size)
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        return {
            'text': tokens,
            'label': label
        }

class WikiText2Dataset(Dataset):
    """WikiText2 dataset implementation using HuggingFace datasets"""
    def __init__(self, split='train', max_seq_len=256, max_vocab_size=None):
        # Load from Hugging Face datasets
        self.dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        self.max_seq_len = max_seq_len
        
        # Collect all text, skipping empty lines
        all_texts = [item['text'] for item in self.dataset if len(item['text'].strip()) > 0]
        
        # Build vocabulary from all text
        counter = Counter()
        for text in all_texts:
            tokens = word_tokenize(text.lower())
            counter.update(tokens)
        
        # Create vocabulary
        self.vocab = Vocab(counter, max_size=max_vocab_size, 
                           specials=('<unk>', '<pad>', '<eos>'))
        
        # Tokenize all text and concatenate
        self.all_tokens = []
        for text in all_texts:
            tokens = word_tokenize(text.lower())
            self.all_tokens.extend(tokens)
            self.all_tokens.append('<eos>')  # End of sentence marker
        
        # Create chunks of max_seq_len tokens
        self.chunks = []
        for i in range(0, len(self.all_tokens) - max_seq_len, max_seq_len):
            chunk = self.all_tokens[i:i + max_seq_len]
            self.chunks.append(chunk)
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        
        # Convert tokens to indices
        indices = [self.vocab[token] for token in chunk]
        
        # Create input and target
        # For language modeling, target is the next token
        input_indices = indices[:-1]
        target_indices = indices[1:]
        
        return {
            'input': input_indices,
            'target': target_indices
        }

def collate_imdb_batch(batch):
    """Collate function for IMDB text classification batches"""
    texts = [item['text'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch])
    
    # Get maximum sequence length in this batch
    max_len = max(len(text) for text in texts)
    
    # Calculate lengths of each text
    lengths = torch.tensor([len(text) for text in texts])
    
    # Pad texts
    padded_texts = []
    for text in texts:
        padded = text + ['<pad>'] * (max_len - len(text))
        padded_texts.append(padded)
    
    return padded_texts, lengths, labels

def collate_wikitext_batch(batch, vocab):
    """Collate function for WikiText2 language modeling batches"""
    inputs = [item['input'] for item in batch]
    targets = [item['target'] for item in batch]
    
    # Get maximum sequence length in this batch
    max_len = max(len(seq) for seq in inputs)
    
    # Pad inputs and targets
    padded_inputs = []
    padded_targets = []
    for input_seq, target_seq in zip(inputs, targets):
        # Pad inputs
        input_padded = input_seq + [vocab['<pad>']] * (max_len - len(input_seq))
        padded_inputs.append(input_padded)
        
        # Pad targets
        target_padded = target_seq + [vocab['<pad>']] * (max_len - len(target_seq))
        padded_targets.append(target_padded)
    
    # Convert to tensors - shape: [batch_size, seq_len]
    input_tensor = torch.tensor(padded_inputs)
    target_tensor = torch.tensor(padded_targets)
    
    # Transformer expects shape [seq_len, batch_size], so transpose tensors
    input_tensor = input_tensor.transpose(0, 1)
    target_tensor = target_tensor.transpose(0, 1)
    
    return input_tensor, target_tensor

def get_imdb_iterators(batch_size, device='cpu'):
    """Get iterators for IMDB dataset"""
    # Create datasets
    train_dataset = IMDBDataset(split='train')
    val_dataset = IMDBDataset(split='val')
    test_dataset = IMDBDataset(split='test')
    
    # Set the vocabulary for validation and test datasets
    val_dataset.vocab = train_dataset.vocab
    test_dataset.vocab = train_dataset.vocab
    
    # Custom collate function
    collate_fn = lambda batch: collate_imdb_batch(batch)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader, train_dataset.vocab

def get_wikitext2_iterators(batch_size, max_seq_len=256, device='cpu'):
    """Get iterators for WikiText2 dataset"""
    # Create datasets
    train_dataset = WikiText2Dataset(split='train', max_seq_len=max_seq_len)
    val_dataset = WikiText2Dataset(split='validation', max_seq_len=max_seq_len)
    test_dataset = WikiText2Dataset(split='test', max_seq_len=max_seq_len)
    
    # Share vocabulary
    val_dataset.vocab = train_dataset.vocab
    test_dataset.vocab = train_dataset.vocab
    
    # Custom collate function
    collate_fn = lambda batch: collate_wikitext_batch(batch, train_dataset.vocab)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader, train_dataset.vocab

def get_tiny_datasets(batch_size, dataset_type, device='cpu'):
    """Get tiny datasets for fast testing/debugging"""
    if dataset_type == 'imdb':
        # Create tiny IMDB dataset
        train_texts = [
            "This movie was great and I loved it",
            "terrible waste of time and money",
            "I loved this film it was amazing",
            "awful acting and direction terrible movie",
            "brilliant screenplay and effects loved it"
        ]
        
        train_labels = [1, 0, 1, 0, 1]  # 1 for positive, 0 for negative
        
        test_texts = [
            "excellent movie experience must watch",
            "one of the worst films ever made",
            "amazing visual effects and story",
            "poor screenplay and acting avoid it"
        ]
        
        test_labels = [1, 0, 1, 0]
        
        # Tokenize
        train_tokenized = [word_tokenize(text.lower()) for text in train_texts]
        test_tokenized = [word_tokenize(text.lower()) for text in test_texts]
        
        # Build vocabulary
        counter = Counter()
        for tokens in train_tokenized:
            counter.update(tokens)
        
        vocab = Vocab(counter, specials=('<unk>', '<pad>'))
        
        # Create datasets
        train_size = int(0.8 * len(train_tokenized))
        train_data = list(zip(train_tokenized[:train_size], train_labels[:train_size]))
        val_data = list(zip(train_tokenized[train_size:], train_labels[train_size:]))
        test_data = list(zip(test_tokenized, test_labels))
        
        # Create collate function
        def tiny_imdb_collate(batch):
            texts = [item[0] for item in batch]
            labels = torch.tensor([item[1] for item in batch])
            
            # Pad texts
            max_len = max(len(text) for text in texts)
            padded_texts = []
            for text in texts:
                padded = text + ['<pad>'] * (max_len - len(text))
                padded_texts.append(padded)
            
            # Convert to indices
            indices = [[vocab[token] for token in text] for text in padded_texts]
            
            return torch.tensor(indices), labels
        
        # Create dataloaders
        train_loader = DataLoader(train_data, batch_size=batch_size, 
                                  shuffle=True, collate_fn=tiny_imdb_collate)
        val_loader = DataLoader(val_data, batch_size=batch_size, 
                                shuffle=False, collate_fn=tiny_imdb_collate)
        test_loader = DataLoader(test_data, batch_size=batch_size, 
                                 shuffle=False, collate_fn=tiny_imdb_collate)
        
        return train_loader, val_loader, test_loader, vocab
        
    elif dataset_type == 'wikitext2':
        # Create tiny WikiText2 dataset
        texts = [
            "The tower is 324 metres tall .",
            "The metal structure weighs 7,300 tonnes .",
            "The tower has three levels for visitors .",
            "Tickets can be purchased to ascend by stairs or lift .",
            "The tower is the most-visited paid monument in the world .",
            "The tower was built as the entrance to the 1889 World's Fair .",
            "It was named after the engineer Gustave Eiffel .",
            "The design of the tower was criticized by many artists .",
            "More than 250 million people have visited the tower since it was completed .",
            "The tower was almost demolished in 1909 .",
            "Today , it is considered a distinctive landmark of Paris ."
        ]
        
        # Split into train/val/test
        train_size = 6
        val_size = 2
        train_texts = texts[:train_size]
        val_texts = texts[train_size:train_size+val_size]
        test_texts = texts[train_size+val_size:]
        
        # Tokenize
        tokenized_texts = [word_tokenize(text.lower()) for text in texts]
        
        # Build vocabulary
        counter = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)
            
        vocab = Vocab(counter, specials=('<unk>', '<pad>', '<eos>'))
        
        # Create datasets with sequences suitable for transformer
        def prepare_lm_data(texts, seq_len=10):
            all_tokens = []
            for tokens in texts:
                all_tokens.extend(tokens)
                all_tokens.append('<eos>')
            
            # Create chunks for easier batch processing
            chunks = []
            for i in range(0, len(all_tokens) - seq_len, seq_len):
                chunk = all_tokens[i:i + seq_len]
                chunks.append(chunk)
            
            # Make sure we have at least one chunk
            if not chunks and all_tokens:
                chunks = [all_tokens]
                
            # For each chunk, create input/target pairs
            pairs = []
            for chunk in chunks:
                input_indices = [vocab[token] for token in chunk[:-1]]
                target_indices = [vocab[token] for token in chunk[1:]]
                
                # Make sure we have at least one input/target pair
                if input_indices and target_indices:
                    pairs.append({
                        'input': input_indices,
                        'target': target_indices
                    })
            
            return pairs
        
        train_pairs = prepare_lm_data([word_tokenize(text.lower()) for text in train_texts])
        val_pairs = prepare_lm_data([word_tokenize(text.lower()) for text in val_texts])
        test_pairs = prepare_lm_data([word_tokenize(text.lower()) for text in test_texts])
        
        # Create custom collate function for transformer data
        def tiny_wikitext_collate(batch):
            inputs = [item['input'] for item in batch]
            targets = [item['target'] for item in batch]
            
            # Get max sequence length in this batch
            max_len = max(len(seq) for seq in inputs)
            
            # Pad sequences
            padded_inputs = []
            padded_targets = []
            
            for input_seq, target_seq in zip(inputs, targets):
                # Pad inputs
                input_padded = input_seq + [vocab['<pad>']] * (max_len - len(input_seq))
                padded_inputs.append(input_padded)
                
                # Pad targets
                target_padded = target_seq + [vocab['<pad>']] * (max_len - len(target_seq))
                padded_targets.append(target_padded)
            
            # Create tensors - shape [batch_size, seq_len]
            input_tensor = torch.tensor(padded_inputs)
            target_tensor = torch.tensor(padded_targets)
            
            # Transpose to shape [seq_len, batch_size] for transformer
            input_tensor = input_tensor.transpose(0, 1)
            target_tensor = target_tensor.transpose(0, 1)
            
            return input_tensor, target_tensor
        
        # Create dataloaders
        train_loader = DataLoader(train_pairs, batch_size=batch_size, 
                                 shuffle=True, collate_fn=tiny_wikitext_collate)
        val_loader = DataLoader(val_pairs, batch_size=batch_size, 
                               shuffle=False, collate_fn=tiny_wikitext_collate)
        test_loader = DataLoader(test_pairs, batch_size=batch_size, 
                                shuffle=False, collate_fn=tiny_wikitext_collate)
        
        return train_loader, val_loader, test_loader, vocab
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}") 