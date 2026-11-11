import os
import json
import argparse
import torch
import numpy as np
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# 💡 모듈화해둔 평가 함수 라이브러리 임포트
from evaluator import (
    eval_mia,
    eval_neuron_activation,
    eval_ppl_and_logprob,
    eval_general_capability
)

# 1.base - open-unlearning/pos_tofu_Llama-3.2-1B-Instruct_full_lr2e-05_wd0.01_epoch10
# 2.retain only - open-unlearning/neg_tofu_Llama-3.2-1B-Instruct_retain90_forget10_pert_lr1e-05_wd0.01_epoch10
# --- [Standard Model 평가 시] ---
_p = argparse.ArgumentParser(description="기준 모델(Base / Retain-only) 절대 지표 평가")
_p.add_argument("--model_name", default="1_base", help="결과 파일 식별 이름 (예: 1_base, 2_retain_only)")
_p.add_argument("--model_path", default="open-unlearning/pos_tofu_Llama-3.2-1B-Instruct_full_lr2e-05_wd0.01_epoch10")
_p.add_argument("--output_dir", default="./eval_standard_results")
_a = _p.parse_args()
MODEL_NAME = _a.model_name
MODEL_PATH = _a.model_path
OUTPUT_DIR = _a.output_dir
# ===========================================================================

MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---------------------------------------------------------------------------
# 1. 모델 로더
# ---------------------------------------------------------------------------
print(f"\n[Baseline Model Load] Loading {MODEL_NAME} from: {MODEL_PATH}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# ---------------------------------------------------------------------------
# 2. TOFU 데이터셋 로드 (평가용 50% Forget 테스트 데이터셋)
# ---------------------------------------------------------------------------
print("[Data Load] Loading TOFU Dataset...")
tofu_forget = list(load_dataset("locuslab/TOFU", "forget10")["train"])
tofu_retain = list(load_dataset("locuslab/TOFU", "retain90")["train"])

split_idx = len(tofu_forget) // 2
test_forget_data = tofu_forget[split_idx:] # 남은 50% 데이터로 평가

forget_texts = [d['question'] + " " + d['answer'] for d in test_forget_data]
retain_texts = [d['question'] + " " + d['answer'] for d in tofu_retain[:len(test_forget_data)]]

# ---------------------------------------------------------------------------
# 3. 기준 모델 순수 지표 추출 (상대 수치 제외)
# ---------------------------------------------------------------------------
print(f"\nEvaluating Baseline Model: [{MODEL_NAME}]...")

metrics_result = {}

# 1) MIA (Forget vs Retain Loss Threshold)
metrics_result.update(eval_mia(model, tokenizer, forget_texts, retain_texts))

# 2) Neuron Activation (중간 레이어 평균 활성화 수치)
metrics_result.update(eval_neuron_activation(model, tokenizer, forget_texts[:20]))

# 3) Perplexity & Log Likelihood (Forget 데이터셋 기준)
ppl_res = eval_ppl_and_logprob(model, tokenizer, forget_texts)
metrics_result["forget_ppl"] = ppl_res["ppl"]
metrics_result["forget_log_prob"] = ppl_res["log_prob"]

# 4) General Capability (MMLU 500개 / GSM8K 200개 적용)
# 💡 tokenizer 인자 추가 및 샘플 수 매개변수 명시적 전달
metrics_result["mmlu_accuracy"] = eval_general_capability(model, tokenizer, benchmark_name="mmlu", num_samples=500)
metrics_result["gsm8k_accuracy"] = eval_general_capability(model, tokenizer, benchmark_name="gsm8k", num_samples=200)

# ---------------------------------------------------------------------------
# 4. 개별 JSON 저장 및 integrated_summary.json 자동 누적/덮어쓰기
# ---------------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 4-1. 개별 파일 저장 (./eval_standard_results/models/{MODEL_NAME}.json)
model_json_path = os.path.join(MODELS_DIR, f"{MODEL_NAME}.json")
model_data = {
    "model_name": MODEL_NAME,
    "model_path": MODEL_PATH,
    "eval_timestamp": timestamp,
    "metrics": metrics_result
}

with open(model_json_path, "w", encoding="utf-8") as f:
    json.dump(model_data, f, indent=4, ensure_ascii=False)
print(f" -> [Individual Saved] {model_json_path}")

# 4-2. 통합 파일 업데이트 (./eval_standard_results/integrated_summary.json)
summary_file_path = os.path.join(OUTPUT_DIR, "integrated_summary.json")

if os.path.exists(summary_file_path):
    with open(summary_file_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)
else:
    summary_data = {"last_updated": timestamp, "models": {}}

summary_data["last_updated"] = timestamp
summary_data["models"][MODEL_NAME] = {
    "model_path": MODEL_PATH,
    "eval_timestamp": timestamp,
    "metrics": metrics_result
}

with open(summary_file_path, "w", encoding="utf-8") as f:
    json.dump(summary_data, f, indent=4, ensure_ascii=False)
print(f" -> [Integrated Summary Updated] {summary_file_path}")

print(f"\nSuccessfully finished baseline evaluation for '{MODEL_NAME}'!")