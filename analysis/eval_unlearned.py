import os
import json
import torch
import numpy as np
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

# 💡 평가 함수 모듈 임포트
from evaluator import (
    eval_mia,
    eval_l2_distance,
    eval_neuron_activation,
    eval_ppl_and_logprob,
    eval_kl_divergence,
    eval_general_capability
)

# ===========================================================================
# 💡 [비교 평가 설정 구역] 내가 비교할 모델 1개만 아래 설정 후 바로 실행!
# ===========================================================================
# 1. 평가 대상 모델 식별 이름 (예: "3_graddiff_fp16", "5_graddiff_quant_4bit", "6_altpo_fp16" 등)
TARGET_MODEL_NAME = "3_graddiff_fp16"

# 2. 평가 대상 모델 경로 (HuggingFace Hub ID 또는 로컬 가중치 디렉토리)
TARGET_MODEL_PATH = "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch5"

# 3. Target 모델의 4-bit 양자화 적용 여부 (True / False)
IS_QUANTIZED = False

# --- [고정 대조군 2종 (Base & Retain-only) 경로] ---
BASE_MODEL_PATH = "open-unlearning/pos_tofu_Llama-3.2-1B-Instruct_full_lr2e-05_wd0.01_epoch10"
RETAIN_MODEL_PATH = "open-unlearning/neg_tofu_Llama-3.2-1B-Instruct_retain90_forget10_pert_lr1e-05_wd0.01_epoch10"

# 결과 저장 루트 디렉토리
OUTPUT_DIR = "./eval_unlearned_results"
# ===========================================================================

MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer 로드 (Base 기준)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---------------------------------------------------------------------------
# 1. 모델 로더 함수
# ---------------------------------------------------------------------------
def load_eval_model(path, is_quant=False):
    print(f"[Model Load] Loading: {path} (Quantized: {is_quant})")
    if is_quant:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb_config, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    return model

# ---------------------------------------------------------------------------
# 2. TOFU 데이터셋 로드 (평가용 50% Forget 테스트 데이터)
# ---------------------------------------------------------------------------
print("[Data Load] Loading TOFU Dataset...")
tofu_forget = list(load_dataset("locuslab/TOFU", "forget10")["train"])
tofu_retain = list(load_dataset("locuslab/TOFU", "retain90")["train"])

split_idx = len(tofu_forget) // 2
test_forget_data = tofu_forget[split_idx:]  # 평가용 50% 테스트셋

forget_texts = [d['question'] + " " + d['answer'] for d in test_forget_data]
retain_texts = [d['question'] + " " + d['answer'] for d in tofu_retain[:len(test_forget_data)]]

# ---------------------------------------------------------------------------
# 3. 평가 대상 모델(Target) 평가 및 Base/Retain-only 1:1 대조
# ---------------------------------------------------------------------------
print(f"\n=======================================================")
print(f" Start 1:1 Comparative Evaluation: [{TARGET_MODEL_NAME}]")
print(f" Target Path: {TARGET_MODEL_PATH}")
print(f"=======================================================")

target_model = load_eval_model(TARGET_MODEL_PATH, IS_QUANTIZED)
metrics_result = {}

# 3-1. Target 모델 고유 지표 계산
print(f"\n1. Calculating Absolute Metrics for [{TARGET_MODEL_NAME}]...")
metrics_result.update(eval_mia(target_model, tokenizer, forget_texts, retain_texts))
metrics_result.update(eval_neuron_activation(target_model, tokenizer, forget_texts[:20]))

ppl_res = eval_ppl_and_logprob(target_model, tokenizer, forget_texts)
metrics_result["forget_ppl"] = ppl_res["ppl"]
metrics_result["forget_log_prob"] = ppl_res["log_prob"]

metrics_result["mmlu_accuracy"] = eval_general_capability(target_model, tokenizer, benchmark_name="mmlu", num_samples=500)
metrics_result["gsm8k_accuracy"] = eval_general_capability(target_model, tokenizer, benchmark_name="gsm8k", num_samples=200)

# 3-2. [대조군 1] Base Model 대비 수치 계산 (_vs_base)
print(f"\n2. Comparing against Baseline 1: [Base Model]...")
base_model = load_eval_model(BASE_MODEL_PATH, is_quant=False)

metrics_result["l2_dist_vs_base"] = eval_l2_distance(base_model, target_model, IS_QUANTIZED)
metrics_result["retain_kl_vs_base"] = eval_kl_divergence(base_model, target_model, tokenizer, retain_texts[:50])

del base_model
torch.cuda.empty_cache()

# 3-3. [대조군 2] Retain-only Model 대비 수치 계산 (_vs_retain)
print(f"\n3. Comparing against Baseline 2: [Retain-only Model]...")
retain_model = load_eval_model(RETAIN_MODEL_PATH, is_quant=False)

metrics_result["l2_dist_vs_retain"] = eval_l2_distance(retain_model, target_model, IS_QUANTIZED)
metrics_result["retain_kl_vs_retain"] = eval_kl_divergence(retain_model, target_model, tokenizer, retain_texts[:50])

# Retain-only 모델의 MIA와 Target 모델 MIA 간의 절대 차이(Gap) 계산
retain_mia = eval_mia(retain_model, tokenizer, forget_texts, retain_texts)["mia_accuracy"]
metrics_result["mia_gap_from_retain"] = float(abs(metrics_result["mia_accuracy"] - retain_mia))

del retain_model
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# 4. 저장 로직 (개별 모델 JSON 저장 및 integrated_summary.json 누적)
# ---------------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 4-1. 개별 파일 저장 (./eval_results/models/{TARGET_MODEL_NAME}.json)
model_json_path = os.path.join(MODELS_DIR, f"{TARGET_MODEL_NAME}.json")
model_data = {
    "model_name": TARGET_MODEL_NAME,
    "model_path": TARGET_MODEL_PATH,
    "base_reference_path": BASE_MODEL_PATH,
    "retain_reference_path": RETAIN_MODEL_PATH,
    "is_quantized": IS_QUANTIZED,
    "eval_timestamp": timestamp,
    "metrics": metrics_result
}

with open(model_json_path, "w", encoding="utf-8") as f:
    json.dump(model_data, f, indent=4, ensure_ascii=False)
print(f"\n -> [Saved Individual Report] {model_json_path}")

# 4-2. 통합 리포트 저장/업데이트 (./eval_results/integrated_summary.json)
summary_file_path = os.path.join(OUTPUT_DIR, "integrated_summary.json")

if os.path.exists(summary_file_path):
    with open(summary_file_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)
else:
    summary_data = {"last_updated": timestamp, "models": {}}

summary_data["last_updated"] = timestamp
summary_data["models"][TARGET_MODEL_NAME] = {
    "model_path": TARGET_MODEL_PATH,
    "eval_timestamp": timestamp,
    "metrics": metrics_result
}

with open(summary_file_path, "w", encoding="utf-8") as f:
    json.dump(summary_data, f, indent=4, ensure_ascii=False)
print(f" -> [Updated Integrated Summary] {summary_file_path}")

print(f"\nCompleted evaluation for '{TARGET_MODEL_NAME}' against Base & Retain-only baselines!")