
# NLP Assignment: Aspect-Term Polarity Classification
CentraleSupélec – NLP Course 2025


Contributors:
- Maria Kontaratou
- Manon Lagarde
- Chaimae Sadoune

---

## Classifier Overview

Our solution uses a fine-tuned transformer-based discriminative model for aspect-based sentiment classification. Specifically, we selected the `facebook/roberta-base` model from the list of allowed Huggingface pre-trained encoder-only models. The model was fine-tuned on a classification head for three sentiment labels: **positive**, **negative**, and **neutral**.

Inputs were formatted as follows to help the model distinguish the focus of sentiment:  
`<aspect> [SEP] <left_context> [SEP] <term> [SEP] <right_context>`  
This format was shown to improve performance on aspect-based classification tasks in previous literature.

The tokenizer and model were loaded using the Huggingface `transformers` library. We used the AdamW optimizer with learning rate scheduling and gradient accumulation to manage memory constraints. Automatic mixed precision (AMP) via `torch.cuda.amp` was used to speed up training and reduce memory consumption.

---

## Class Imbalancing Handling
During the Exploratory Data Analysis (EDA), we observed **a significant imbalance** in label distribution:  
- 70% of examples were labeled as **positive**,  
- 26% as **negative**,  
- and only 4% as **neutral**.

To address this imbalance, we implemented class weighting using the **inverse normalized frequency** of each class:  
`[0.4755, 8.6458, 1.8787]` (corresponding to positive, neutral, and negative).  
These weights were applied to the cross-entropy loss function, helping the model focus more on the minority classes without overfitting.

---

## Technical Overview and Model Implementation
1. Model Choice and Initialization
We implemented a transformer-based classifier using the facebook/roberta-base model from the Hugging Face library, which is part of the authorized set of encoder-only models. It was selected because it offers a favorable balance between accuracy and execution speed, making it suitable for the evaluation environment’s constraints (e.g., <14GB GPU memory, max ~5 mins execution per run).

Initialization:

```bash
self.model_name = 'roberta-base'
self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
self.model = RobertaForSequenceClassification.from_pretrained(self.model_name, num_labels=3)` 

2. Preprocessing Strategy
Input samples were formatted to clearly separate different parts of the sentence:

```bash
text = f"{aspect} [SEP] {left} [SEP] {target} [SEP] {right}"
This structure helps the model identify the target term, its surrounding context, and the aspect category, all of which are crucial for accurate aspect-term polarity classification.

3. Addressing Label Imbalance (Discovered in EDA)
During our exploratory data analysis (EDA), we examined the label distribution in both traindata.csv and devdata.csv and found a significant imbalance:

Positive: ~70%

Negative: ~26%

Neutral: ~4%

This imbalance would bias the model toward predicting the majority class, thus hurting performance on minority classes, especially neutral.

To counter this, we computed inverse label frequency weights and normalized them to avoid over-scaling. The weights used were:

```bash
class_weights = torch.tensor([0.4755, 8.6458, 1.8787]).to(device)
These were applied through PyTorch’s CrossEntropyLoss:

```bash
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
This approach improved model sensitivity to the underrepresented neutral and negative classes.

4. Training Optimization for Efficiency
To ensure training remains both accurate and efficient:

Batch size: 16

Gradient accumulation steps: 2
→ Simulates a larger batch size while fitting into the memory limit.

Max sequence length: 192
→ Long enough to preserve context, short enough for fast execution.

Training epochs: 15
→ Balanced with early stopping via best model checkpointing.

Mixed precision training (AMP) was applied for:

Faster computation

Reduced memory usage

```bash
with autocast():
    outputs = self.model(input_ids, attention_mask=attn_mask)
    loss = loss_fn(outputs.logits, labels) / self.gradient_accumulation_steps
This allowed us to stay within the ~4.5-minute window per run, without compromising accuracy.

5. Evaluation Strategy
After each epoch, the model was evaluated on the development set. The best-performing model across epochs (based on dev accuracy) was retained for final predictions.

This mechanism avoided overfitting and ensured we didn't rely on a specific epoch, making the system more robust.

## Summary of results:

Model               | Mean Dev Acc. | Std Dev | Time per run (s)
--------------------|---------------|---------|------------------
roberta-base        | 87.45%        | 0.39    | 265
deberta-v3-base     | 84.58%        | 0.87    | 261
roberta-large       | 89.58%        | 0.64    | 651
deberta-v3-large    | 89.31%        | 1.18    | 1072

## Requirements
pip install transformers==4.50.3 peft==0.15.1 trl==0.16.0 datasets==3.5.0 \
sentencepiece==0.2.0 lightning==2.5.1 ollama==0.4.7 pyrallis==0.3.1 torch==2.6.0
