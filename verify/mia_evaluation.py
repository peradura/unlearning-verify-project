import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from datasets import load_dataset
import json
from tqdm import tqdm
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Black-box MIA Evaluation for Unlearned LLM")
    parser.add_argument("--model_path", type=str, required=True, help="HuggingFace model path or local directory")
    parser.add_argument("--forget_data", type=str, required=True, help="Forget set (jsonl or HF dataset)")
    parser.add_argument("--nonmember_data", type=str, required=True, help="Non-member set")
    parser.add_argument("--max_samples", type=int, default=500, help="최대 샘플 수 (각 set)")
    parser.add_argument("--max_length", type=int, default=512, help="최대 토큰 길이")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--method", type=str, choices=["loss", "mink", "both"], default="both")
    parser.add_argument("--k", type=float, default=0.2, help="Min-K%에서 사용할 k (0.0~1.0)")
    parser.add_argument("--output", type=str, default="mia_results.json")
    return parser.parse_args()

def load_model_and_tokenizer(model_path):
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer

def compute_loss(model, tokenizer, texts, max_length=512):
    """Average per-token negative log-likelihood"""
    losses = []
    with torch.no_grad():
        for text in tqdm(texts, desc="Computing loss"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
            
            if inputs.input_ids.shape[1] < 10:  # 너무 짧으면 skip
                continue
                
            labels = inputs.input_ids.clone()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss.item()  # average NLL
            losses.append(loss)
    return np.array(losses)

def compute_mink_prob(model, tokenizer, texts, k=0.2, max_length=512):
    """Min-K% Prob: k% lowest probability tokens의 평균 log prob"""
    scores = []
    with torch.no_grad():
        for text in tqdm(texts, desc="Computing Min-K%"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
            if inputs.input_ids.shape[1] < 20:
                continue
                
            outputs = model(**inputs)
            logits = outputs.logits
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # next-token log probs
            target_log_probs = log_probs[0, :-1, :].gather(1, inputs.input_ids[0, 1:].unsqueeze(1)).squeeze(1)
            
            # k% lowest (가장 작은 = 가장 outlier)
            k_count = max(1, int(len(target_log_probs) * k))
            min_k_logprobs = torch.topk(target_log_probs, k_count, largest=False).values
            score = min_k_logprobs.mean().item()
            scores.append(score)
    return np.array(scores)

def load_data(path, max_samples=500):
    """
    [BUG FIX]
    원본: data = [json.loads(line) if ... else json.load(f)]
      → .jsonl의 경우 'line' 변수가 정의되지 않아 NameError 발생
      → .json의 경우 전체 데이터를 리스트로 한 번 더 감싸 1개만 읽힘
    수정: jsonl은 모든 라인을 순회, json은 감싸지 않고 직접 load
    """
    if path.endswith('.jsonl'):
        with open(path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]
        texts = [item['text'] for item in data[:max_samples] if 'text' in item]

    elif path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            texts = [item['text'] for item in data[:max_samples] if isinstance(item, dict) and 'text' in item]
        else:
            texts = [data['text']] if 'text' in data else []

    else:
        # HuggingFace dataset
        ds = load_dataset(path, split='train')
        texts = ds['text'][:max_samples] if 'text' in ds.column_names else ds[:max_samples]

    return texts

def main():
    args = parse_args()
    
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    # 데이터 로드
    forget_texts = load_data(args.forget_data, args.max_samples)
    nonmember_texts = load_data(args.nonmember_data, args.max_samples)
    
    print(f"Forget samples: {len(forget_texts)}, Non-member: {len(nonmember_texts)}")
    
    results = {"model": args.model_path}
    
    # Loss-based
    if args.method in ["loss", "both"]:
        print("\n=== Loss-based MIA ===")
        forget_loss = compute_loss(model, tokenizer, forget_texts, args.max_length)
        non_loss = compute_loss(model, tokenizer, nonmember_texts, args.max_length)
        
        y_true = np.concatenate([np.ones(len(forget_loss)), np.zeros(len(non_loss))])
        y_score = np.concatenate([-forget_loss, -non_loss])  # 낮은 loss = member (score 높게)
        
        auc = roc_auc_score(y_true, y_score)
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        y_pred = (y_score >= optimal_threshold).astype(int)
        acc = accuracy_score(y_true, y_pred)
        
        results["loss"] = {"auc": float(auc), "accuracy": float(acc), "optimal_threshold": float(optimal_threshold)}
        print(f"Loss AUC: {auc:.4f} (50%에 가까울수록 성공)")
        print(f"Accuracy: {acc:.4f}")
    
    # Min-K%
    if args.method in ["mink", "both"]:
        print("\n=== Min-K% Prob ===")
        forget_mink = compute_mink_prob(model, tokenizer, forget_texts, args.k, args.max_length)
        non_mink = compute_mink_prob(model, tokenizer, nonmember_texts, args.k, args.max_length)
        
        y_true = np.concatenate([np.ones(len(forget_mink)), np.zeros(len(non_mink))])
        y_score = np.concatenate([forget_mink, non_mink])  # Min-K score가 낮을수록 member
        
        auc = roc_auc_score(y_true, -y_score)  # 낮을수록 member
        results["mink"] = {"auc": float(auc), "k": args.k}
        print(f"Min-K% AUC: {auc:.4f}")
    
    # 결과 저장
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n결과 저장: {args.output}")
    print("→ AUC ≈ 0.5 이면 unlearning이 잘 된 것!")

if __name__ == "__main__":
    main()
