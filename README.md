
NLP Assignment: Aspect-Term Polarity Classification
CentraleSupélec – NLP Course 2025


Contributors:
- Maria Kontaratou
- Manon Lagarde
- Chaimae Sadoune

---

1. CLASSIFIER OVERVIEW

Our solution uses a fine-tuned transformer-based discriminative model for aspect-based sentiment classification. Specifically, we selected the `facebook/roberta-base` model from the list of allowed Huggingface pre-trained encoder-only models. The model was fine-tuned on a classification head for three sentiment labels: **positive**, **negative**, and **neutral**.

Inputs were formatted as follows to help the model distinguish the focus of sentiment:  
`<aspect> [SEP] <left_context> [SEP] <term> [SEP] <right_context>`  
This format was shown to improve performance on aspect-based classification tasks in previous literature.

The tokenizer and model were loaded using the Huggingface `transformers` library. We used the AdamW optimizer with learning rate scheduling and gradient accumulation to manage memory constraints. Automatic mixed precision (AMP) via `torch.cuda.amp` was used to speed up training and reduce memory consumption.

---

2. CLASS IMBALANCE HANDLING

During the Exploratory Data Analysis (EDA), we observed **a significant imbalance** in label distribution:  
- 70% of examples were labeled as **positive**,  
- 26% as **negative**,  
- and only 4% as **neutral**.

To address this imbalance, we implemented class weighting using the **inverse normalized frequency** of each class:  
`[0.4755, 8.6458, 1.8787]` (corresponding to positive, neutral, and negative).  
These weights were applied to the cross-entropy loss function, helping the model focus more on the minority classes without overfitting.

---

3. IMPLEMENTATION DETAILS

We used the following configuration for our final model (`facebook/roberta-base`):

- `batch_size`: 16  
- `max_length`: 192  
- `epochs`: 15  
- `gradient_accumulation_steps`: 2  
- `learning_rate`: 2e-5  
- `loss_function`: CrossEntropy with class weights  
- `optimizer`: AdamW  
- `device`: passed as parameter (no hard-coded device)

All pre-processing was integrated inside the `train()` and `predict()` methods, as required by the assignment constraints. We strictly followed the structure of `classifier.py` and did not modify `tester.py`.

Required libraries (installed with exact versions as per assignment instructions):

```bash
pip install transformers==4.50.3 peft==0.15.1 trl==0.16.0 datasets==3.5.0 \
sentencepiece==0.2.0 lightning==2.5.1 ollama==0.4.7 pyrallis==0.3.1 torch==2.6.0
