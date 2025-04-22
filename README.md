
# NLP Assignment: Aspect-Term Polarity Classification
CentraleSupélec – NLP Course 2025


Contributors:
- Maria Kontaratou
- Manon Lagarde
- Chaimae Sadoune

## Classifier Overview

Our solution uses a fine-tuned transformer-based discriminative model specifically tailored for aspect-based sentiment classification. After evaluating several models from the authorized list provided, we chose the `facebook/roberta-base` model due to its optimal balance between accuracy, computational efficiency, and GPU memory usage. The model was fine-tuned on a classification head for three sentiment labels: **positive**, **negative**, and **neutral**.

The model inputs were specifically formatted as follows to distinctly highlight the sentiment-bearing term and its relevant context: 
`<aspect> [SEP] <left_context> [SEP] <term> [SEP] <right_context>`  
This structured input explicitly segments the aspect, the targeted sentiment term, and the contextual text surrounding the term. Previous studies have demonstrated that clearly delimiting the context and target term significantly improves performance in fine-grained sentiment analysis tasks.

We utilized Huggingface’s `transformers` library to load and tokenize data. Optimization techniques included AdamW with linear learning rate scheduling and gradient accumulation to mitigate GPU memory constraints. We also implemented Automatic Mixed Precision (AMP) training (`torch.cuda.amp`), significantly reducing memory requirements and computational time while preserving model performance.


## Class Imbalance Handling (Identified in Exploratory Analysis)
Through an extensive Exploratory Data Analysis (EDA), our team identified a critical challenge—the dataset exhibited a severe imbalance among sentiment labels: 
- 70% of examples were labeled as **positive**
- 26% as **negative**
- and only 4% as **neutral**


This disproportionate representation of sentiment classes posed a high risk of bias, potentially leading the classifier to predominantly predict the majority class (positive sentiment), thus adversely affecting the classification accuracy for minority classes (negative and neutral).

To mitigate this bias, we adopted a strategy of applying **inverse normalized class frequency weighting** to the loss function during training. Specifically, we calculated the normalized inverse frequencies of the classes as:
`[0.4755, 8.6458, 1.8787]` (corresponding to positive, neutral, and negative)
These weights were applied to the cross-entropy loss function, helping the model focus more on the minority classes without overfitting.


## Technical Overview and Model Implementation
### Model Choice and Initialization
After comparative testing with several authorized models, including deberta-v3-base, roberta-large, and deberta-v3-large, we chose the roberta-base model due to its excellent trade-off between accuracy (87.45%) and runtime efficiency (~4.5-5 minutes per run). This model was initialized as follows:


```python
self.model_name = 'roberta-base'
self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
self.model = RobertaForSequenceClassification.from_pretrained(self.model_name, num_labels=3)
```

### Preprocessing Strategy
To clearly delineate the term under analysis and its context, we adopted a structured input format for the tokenizer:
```python
text = f"{aspect} [SEP] {left} [SEP] {target} [SEP] {right}"
```
This explicit segmentation enables the transformer model to precisely pinpoint the sentiment focus and associated context, crucial for aspect-term polarity tasks.


### Optimization Techniques and Hyperparameters
Extensive empirical testing and analysis led us to choose the following parameters for optimal accuracy and resource efficiency:


- Batch size: 16 (optimal for GPU memory constraints)
- Gradient accumulation steps: 2 (effectively simulating a batch size of 32, minimizing memory usage)
- Max sequence length: 192 tokens (carefully selected to ensure adequate context is captured without unnecessary computational overhead)

- Epochs: 15, balanced with early stopping via best epoch checkpointing, ensuring robust generalization

To enhance computational efficiency and significantly reduce training time, we implemented automatic mixed precision (AMP):

```python
with autocast():
    outputs = self.model(input_ids, attention_mask=attn_mask)
    loss = loss_fn(outputs.logits, labels) / self.gradient_accumulation_steps
```
This reduced per-run execution times to under 5 minutes, comfortably meeting the assignment constraints without compromising the quality of predictions.

### Evaluation Strategy
Our training loop featured rigorous evaluation after each epoch on the development dataset (devdata.csv). This allowed us to perform model checkpointing—saving only the model state with the best accuracy on the dev set. Such an approach prevented overfitting, improved robustness, and ensured optimal final performance on unseen data.


## Analytical Overview of Model Selection and Final Results
Throughout our experimentation phase, we conducted rigorous testing on several models. The detailed empirical results are as follows:

Model               | Mean Dev Acc. | Std Dev | Time per run (s)
--------------------|---------------|---------|------------------
roberta-base        | 87.45%        | 0.39    | 265
deberta-v3-base     | 84.58%        | 0.87    | 261
roberta-large       | 89.58%        | 0.64    | 651
deberta-v3-large    | 89.31%        | 1.18    | 1072

Given the constraints of the project—specifically, GPU memory usage (<14GB), strict runtime requirements (~5 minutes per run), and the necessity of consistent performance—we concluded the roberta-base model provided the optimal balance. While larger models (roberta-large, deberta-v3-large) achieved slightly higher accuracy, their significantly greater computational demands (execution times 2 to 4 times longer) outweighed marginal accuracy improvements for our scenario.

## Requirements
```python
pip install transformers==4.50.3 peft==0.15.1 trl==0.16.0 datasets==3.5.0 \
sentencepiece==0.2.0 lightning==2.5.1 ollama==0.4.7 pyrallis==0.3.1 torch==2.6.0
```
