import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.metrics import precision_score, f1_score
import Levenshtein as lev
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # Import BLEU score

# Model name (using RoBERTa model)
model_name = "aubmindlab/bert-base-arabert"

# Load pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load the RoBERTa model for masked language modeling
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Example sentences for evaluation
with open('/home/usd.local/kanishka.parankusham/rizk_lab/shared/kanishka/dataset/para_ara.txt', 'r', encoding='utf-8') as file:
    eval_sentences = [line.strip() for line in file][:100]  # Limit to 100 sentences for efficiency

# Tokenize the sentences, handle truncation
max_length = 512  # Set to a specific value for truncation (e.g., 512)
encoded_sentences = tokenizer(eval_sentences, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
input_ids = encoded_sentences["input_ids"]
attention_mask = encoded_sentences["attention_mask"]

# Replace a random 15% of the tokens with the mask token
masked_input_ids = input_ids.clone()
indices_to_mask = torch.bernoulli(torch.full(input_ids.shape, 0.15)).bool() & (input_ids != tokenizer.pad_token_id)
masked_input_ids[indices_to_mask] = tokenizer.mask_token_id

# Get outputs from the model
outputs = model(masked_input_ids, attention_mask=attention_mask)

# Access logits
logits = outputs.logits

# Calculate predictions
predicted_ids = torch.argmax(logits, dim=-1)

# Calculate accuracy and other metrics
masked_positions = indices_to_mask.sum().item()

if masked_positions > 0:
    correct_predictions = torch.eq(predicted_ids[indices_to_mask], input_ids[indices_to_mask]).sum().item()
    accuracy = (correct_predictions / masked_positions) * 100
    
    # Calculate Precision, Recall, and F1 Score
    y_true = input_ids[indices_to_mask].cpu().numpy()
    y_pred = predicted_ids[indices_to_mask].cpu().numpy()

    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Calculate MRR
    reciprocal_ranks = []
    for true_id, pred_id in zip(y_true, y_pred):
        reciprocal_ranks.append(1 if true_id == pred_id else 0)
    
    mrr = np.mean(reciprocal_ranks)
    
    # Calculate CER
    true_tokens = tokenizer.batch_decode(y_true, skip_special_tokens=True)
    pred_tokens = tokenizer.batch_decode(y_pred, skip_special_tokens=True)

    cer_scores = []
    for true, pred in zip(true_tokens, pred_tokens):
        if len(true) > 0 or len(pred) > 0:
            cer_score = lev.distance(true, pred) / max(len(true), len(pred))
        else:
            cer_score = 0.0
        cer_scores.append(cer_score)

    cer = np.mean(cer_scores) if cer_scores else float('inf')

    # Calculate Hit@K
    k = 10
    top_k_predictions = torch.topk(logits, k, dim=-1).indices
    hits_at_k = np.mean([1 if true in pred[:k] else 0 for true, pred in zip(y_true, top_k_predictions.cpu().numpy())])

    # Calculate BLEU score
    bleu_scores = []
    smoothing_function = SmoothingFunction().method1  # Use a smoothing function for better results
    for true, pred in zip(true_tokens, pred_tokens):
        reference = [true.split()]  # Reference should be a list of lists
        candidate = pred.split()
        bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothing_function)
        bleu_scores.append(bleu_score)

    bleu = np.mean(bleu_scores) if bleu_scores else 0.0

    # Calculate Perplexity
    # Softmax to get probabilities
    softmax_probs = torch.softmax(logits.view(-1, logits.size(-1)), dim=-1)

    # Gather the probabilities of the true tokens
    true_probs = softmax_probs[torch.arange(softmax_probs.size(0)), input_ids.view(-1)].reshape(input_ids.size())

    # Check if there are any zeros in true_probs
    if (true_probs == 0).any():
        print("Warning: Some true probabilities are zero.")

    # Calculate the negative log likelihood
    neg_log_likelihood = -torch.log(true_probs + 1e-10)  # Add a small value to avoid log(0)

    # Handle cases where masked_positions is zero or where negative log likelihood is infinite
    perplexity = torch.exp(neg_log_likelihood.sum() / masked_positions).item() if masked_positions > 0 else float('inf')

else:
    correct_predictions = 0
    accuracy = 0.0
    precision = 0.0
    f1 = 0.0
    mrr = 0.0
    cer = float('inf')
    hits_at_k = 0.0
    bleu = 0.0
    perplexity = float('inf')  # Undefined if there are no masked positions

print("Number of correct predictions:", correct_predictions)
print("Total masked positions:", masked_positions)
print("Accuracy:", accuracy)
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
print(f"Character Error Rate (CER): {cer:.4f}")
print(f"Hit@{k}: {hits_at_k:.4f}")
print(f"BLEU Score: {bleu:.4f}")
print(f"Perplexity: {perplexity:.4f}")
print(f"Masked Language Modeling Accuracy: {accuracy:.4f}%")



