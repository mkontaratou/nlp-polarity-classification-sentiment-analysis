from typing import List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
import pandas as pd
import numpy as np
from torch.cuda.amp import autocast, GradScaler


class Classifier:
    """
    The Classifier: complete the definition of this class template by completing the __init__() function and
    the 2 methods train() and predict() below. Please do not change the signature of these methods
     """


    ############################################# complete the classifier class below
    

    def __init__(self, ollama_url: str):
        """
        This should create and initilize the model.
        !!!!! If the approach you have choosen is in-context-learning with an LLM from Ollama, you should initialize
         the ollama client here using the 'ollama_url' that is provided (please do not use your own ollama
         URL!)
        !!!!! If you have choosen an approach based on training an MLM or a generative LM, then your model should
        be defined and initialized here.
        """
                
        self.model_name = 'roberta-base'
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(self.model_name, num_labels=3)
        self.batch_size = 16
        self.max_length = 192
        self.epochs = 15
        self.gradient_accumulation_steps = 2





    
    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
        If the approach you have choosen is in-context-learning with an LLM from Ollama, you must
          not train the model, and this method should contain only the "pass" instruction
        Otherwise:
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS

        """
        label_map = {'positive': 0, 'neutral': 1, 'negative': 2}

        # === Load and process training data ===
        train_df = pd.read_csv(train_filename, sep="\t", header=None,
                               names=["label", "aspect", "term", "offset", "sentence"])
        train_input_ids, train_attention_masks, train_labels = [], [], []

        for _, row in train_df.iterrows():
            start, end = map(int, row["offset"].split(":"))
            left = row["sentence"][:start].strip()
            target = row["term"]
            right = row["sentence"][end:].strip()
            aspect = row["aspect"]
            label = label_map[row["label"]]

            text = f"{aspect} [SEP] {left} [SEP] {target} [SEP] {right}"
            encoded = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
            train_input_ids.append(encoded["input_ids"].squeeze(0))
            train_attention_masks.append(encoded["attention_mask"].squeeze(0))
            train_labels.append(torch.tensor(label))

        train_dataset = TensorDataset(
            torch.stack(train_input_ids),
            torch.stack(train_attention_masks),
            torch.stack(train_labels)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # === Dev set ===
        dev_df = pd.read_csv(dev_filename, sep="\t", header=None,
                             names=["label", "aspect", "term", "offset", "sentence"])
        dev_input_ids, dev_attention_masks, dev_labels = [], [], []

        for _, row in dev_df.iterrows():
            start, end = map(int, row["offset"].split(":"))
            left = row["sentence"][:start].strip()
            target = row["term"]
            right = row["sentence"][end:].strip()
            aspect = row["aspect"]
            label = label_map[row["label"]]

            text = f"{aspect} [SEP] {left} [SEP] {target} [SEP] {right}"
            encoded = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
            dev_input_ids.append(encoded["input_ids"].squeeze(0))
            dev_attention_masks.append(encoded["attention_mask"].squeeze(0))
            dev_labels.append(torch.tensor(label))

        dev_dataset = TensorDataset(
            torch.stack(dev_input_ids),
            torch.stack(dev_attention_masks),
            torch.stack(dev_labels)
        )
        dev_loader = DataLoader(dev_dataset, batch_size=self.batch_size)

        # === Model Training ===
        self.model.to(device)
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        total_steps = len(train_loader) * self.epochs // self.gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)
        scaler = GradScaler()

        class_weights = torch.tensor([0.4755, 8.6458, 1.8787]).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        best_model_state = None
        best_acc = 0.0

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            optimizer.zero_grad()

            for i, batch in enumerate(train_loader):
                input_ids, attn_mask, labels = [b.to(device) for b in batch]

                with autocast():
                    outputs = self.model(input_ids, attention_mask=attn_mask)
                    loss = loss_fn(outputs.logits, labels) / self.gradient_accumulation_steps

                scaler.scale(loss).backward()
                total_loss += loss.item()

                if (i + 1) % self.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

            # === Evaluation ===
            self.model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch in dev_loader:
                    input_ids, attn_mask, labels = [b.to(device) for b in batch]
                    outputs = self.model(input_ids, attention_mask=attn_mask)
                    preds = torch.argmax(outputs.logits, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            acc = correct / total
            print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Dev Accuracy={acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_model_state = self.model.state_dict()

        if best_model_state:
            self.model.load_state_dict(best_model_state)



    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
        If the approach you have choosen is in-context-learning with an LLM from Ollama, ignore the 'device'
        parameter (because the device is specified when launching the Ollama server, and not by the client side)
        Otherwise:
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """
        label_map = {0: 'positive', 1: 'neutral', 2: 'negative'}
        df = pd.read_csv(data_filename, sep="\t", header=None,
                         names=["label", "aspect", "term", "offset", "sentence"])
        input_ids, attention_masks = [], []

        for _, row in df.iterrows():
            start, end = map(int, row["offset"].split(":"))
            left = row["sentence"][:start].strip()
            target = row["term"]
            right = row["sentence"][end:].strip()
            aspect = row["aspect"]

            text = f"{aspect} [SEP] {left} [SEP] {target} [SEP] {right}"
            encoded = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
            input_ids.append(encoded["input_ids"].squeeze(0))
            attention_masks.append(encoded["attention_mask"].squeeze(0))

        dataset = TensorDataset(torch.stack(input_ids), torch.stack(attention_masks), torch.zeros(len(input_ids)))
        loader = DataLoader(dataset, batch_size=self.batch_size)

        self.model.to(device)
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in loader:
                input_ids, attn_mask, _ = [b.to(device) for b in batch]
                outputs = self.model(input_ids, attention_mask=attn_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend([label_map[p.item()] for p in preds])

        return predictions






