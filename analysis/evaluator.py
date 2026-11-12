import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import re
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1. MIA (Membership Inference Attack Accuracy)
# ---------------------------------------------------------------------------
def eval_mia(model, tokenizer, forget_texts, retain_texts):
    if model is None: return None
    forget_losses, retain_losses = [], []
    with torch.no_grad():
        for text in forget_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            inputs['labels'] = inputs['input_ids']
            forget_losses.append(model(**inputs).loss.item())

        for text in retain_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            inputs['labels'] = inputs['input_ids']
            retain_losses.append(model(**inputs).loss.item())

    threshold = np.mean(retain_losses)
    acc = (sum(l < threshold for l in forget_losses) + sum(l >= threshold for l in retain_losses)) / (
            len(forget_losses) + len(retain_losses))
    return {"mia_accuracy": float(acc)}


# ---------------------------------------------------------------------------
# 2. L2 Distance (두 모델 파라미터 간 거리)
# ---------------------------------------------------------------------------
def eval_l2_distance(model_a, model_b, is_quantized=False):
    if is_quantized or model_a is None or model_b is None:
        return -1.0
    l2_sum = 0.0
    for (n1, p1), (n2, p2) in zip(model_a.named_parameters(), model_b.named_parameters()):
        if p1.requires_grad and p1.dtype == p2.dtype:
            l2_sum += torch.sum((p1.data - p2.data.to(p1.device)) ** 2).item()
    return float(np.sqrt(l2_sum))


# ---------------------------------------------------------------------------
# 3. Neuron Activation (지정 레이어 평균 활성화 수치)
# ---------------------------------------------------------------------------
NEURON_LAYER_IDX = 5  # Llama-3.2-1B(16 layers) 기준 5번 레이어 (2026-11-12, 기존 8번에서 변경)


def eval_neuron_activation(model, tokenizer, sample_texts, layer_idx=NEURON_LAYER_IDX):
    if model is None: return None
    activations = []

    def hook_fn(module, input, output):
        activations.append(output[0].abs().mean().item())

    target_layer = model.model.layers[layer_idx]
    handle = target_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        for text in sample_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            model(**inputs)

    handle.remove()
    return {"mean_neuron_activation": float(np.mean(activations)), "neuron_layer_idx": int(layer_idx)}


# ---------------------------------------------------------------------------
# 4. Perplexity & Log Probability
# ---------------------------------------------------------------------------
def eval_ppl_and_logprob(model, tokenizer, eval_texts):
    if model is None: return {"ppl": None, "log_prob": None}
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for text in eval_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            inputs['labels'] = inputs['input_ids'].clone()
            outputs = model(**inputs)
            loss = outputs.loss.item()
            num_tokens = inputs['input_ids'].size(1)

            total_loss += loss * num_tokens
            total_tokens += num_tokens

    avg_log_prob = -(total_loss / total_tokens) if total_tokens > 0 else 0
    ppl = float(np.exp(total_loss / total_tokens)) if total_tokens > 0 else 0
    return {"ppl": ppl, "log_prob": float(avg_log_prob)}


# ---------------------------------------------------------------------------
# 5. KL Divergence (두 모델간 출력 분포 차이)
# ---------------------------------------------------------------------------
def eval_kl_divergence(model_source, model_target, tokenizer, eval_texts):
    if model_source is None or model_target is None: return None
    kl_list = []
    with torch.no_grad():
        for text in eval_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            p_logits = model_source(**inputs).logits
            q_logits = model_target(**inputs).logits

            p_probs = F.softmax(p_logits, dim=-1)
            q_log_probs = F.log_softmax(q_logits, dim=-1)

            kl = F.kl_div(q_log_probs, p_probs, reduction='batchmean')
            kl_list.append(kl.item())
    return float(np.mean(kl_list))


# ---------------------------------------------------------------------------
# 6. MMLU 실측 평가 (Logprob 500개 샘플)
# ---------------------------------------------------------------------------
def eval_mmlu(model, tokenizer, num_samples=500):
    if model is None: return None
    print(f" -> [MMLU Evaluation] Running {num_samples} samples...")

    ds = load_dataset("cais/mmlu", "all", split="test")
    if num_samples and num_samples < len(ds):
        ds = ds.select(range(num_samples))  # 고정 슬라이싱 (순서 보장)

    choices = ["A", "B", "C", "D"]
    choice_ids = [tokenizer.encode(c, add_special_tokens=False)[-1] for c in choices]

    correct = 0
    with torch.no_grad():
        for item in ds:
            prompt = f"Question: {item['question']}\nA) {item['choices'][0]}\nB) {item['choices'][1]}\nC) {item['choices'][2]}\nD) {item['choices'][3]}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            logits = model(**inputs).logits[0, -1]
            choice_logits = logits[choice_ids]
            pred = torch.argmax(choice_logits).item()

            if pred == item['answer']:
                correct += 1

    acc = correct / len(ds)
    return float(acc)


# ---------------------------------------------------------------------------
# 7. GSM8K 실측 평가 (텍스트 생성 & 숫자 매칭 200개 샘플)
# ---------------------------------------------------------------------------
def _normalize_number(s):
    """'1,234.' / '20.00' / '42' 를 모두 같은 표준형으로 (문장 끝 마침표·천단위 콤마·소수점 0 제거)"""
    s = s.replace(",", "").strip().rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def eval_gsm8k(model, tokenizer, num_samples=200):
    if model is None: return None
    print(f" -> [GSM8K Evaluation] Running {num_samples} samples...")

    ds = load_dataset("openai/gsm8k", "main", split="test")
    if num_samples and num_samples < len(ds):
        ds = ds.select(range(num_samples))  # 고정 슬라이싱 (순서 보장)

    correct = 0
    with torch.no_grad():
        for item in ds:
            prompt = f"Question: {item['question']}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False  # 결정론적 생성 (Greedy Decoding)
            )
            generated_text = tokenizer.decode(output_ids[0][inputs.input_ids.size(1):], skip_special_tokens=True)

            target_match = re.search(r"####\s*(-?\d+[\d,.]*)", item['answer'])
            target_num = _normalize_number(target_match.group(1)) if target_match else ""

            pred_nums = re.findall(r"-?\d+[\d,.]*", generated_text)
            pred_num = _normalize_number(pred_nums[-1]) if pred_nums else ""

            if target_num and pred_num and target_num == pred_num:
                correct += 1

    acc = correct / len(ds)
    return float(acc)


# 통합 인터페이스 Wrapper
def eval_general_capability(model, tokenizer, benchmark_name="mmlu", num_samples=None):
    if benchmark_name.lower() == "mmlu":
        n_samples = num_samples if num_samples is not None else 500
        return eval_mmlu(model, tokenizer, num_samples=n_samples)
    elif benchmark_name.lower() == "gsm8k":
        n_samples = num_samples if num_samples is not None else 200
        return eval_gsm8k(model, tokenizer, num_samples=n_samples)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")