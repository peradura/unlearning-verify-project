import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ===========================================================================
# 💡 [설정 구역] 양자화할 언러닝 모델 경로와 저장할 디렉토리 설정
# ===========================================================================
# 1. 양자화할 대상 언러닝 모델 (HF Hub ID 또는 로컬 FP16 가중치 디렉토리)
# 예시:
# - GradDiff: "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch5"
# - AltPO:    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.1_alpha1_epoch10"
# - NPO:      "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr2e-05_beta0.5_alpha1_epoch10"
_p = argparse.ArgumentParser(description="언러닝 모델 BitsAndBytes 양자화")
_p.add_argument("--model_path", default="open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch5")
_p.add_argument("--save_dir", default="./checkpoints/5_graddiff_quant_4bit")
_p.add_argument("--bits", type=int, choices=[4, 8], default=4)
_a = _p.parse_args()
MODEL_PATH = _a.model_path
QUANT_BITS = _a.bits
SAVE_DIR = _a.save_dir
# ===========================================================================

print(f"=======================================================")
print(f" Start Model Quantization ({QUANT_BITS}-bit)")
print(f" Source Model: {MODEL_PATH}")
print(f" Output Path : {SAVE_DIR}")
print(f"=======================================================")

# 디렉토리 생성
os.makedirs(SAVE_DIR, exist_ok=True)

# 1. Tokenizer 로드 및 저장
print("\n[1/3] Loading and saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.save_pretrained(SAVE_DIR)

# 2. BitsAndBytes Quantization Config 설정
print(f"\n[2/3] Configuring BitsAndBytes {QUANT_BITS}-bit quantization...")
if QUANT_BITS == 4:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",               # NF4 (Normal Float 4) 방식
        bnb_4bit_compute_dtype=torch.float16,     # 연산 시 float16 사용
        bnb_4bit_use_double_quant=True            # 메모리 추가 절약을 위한 이중 양자화
    )
elif QUANT_BITS == 8:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
else:
    raise ValueError("QUANT_BITS must be either 4 or 8.")

# 3. 모델 로드 (양자화 적용)
print(f"[Model Load] Loading target unlearned model into GPU with {QUANT_BITS}-bit precision...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto"
)

# 4. 양자화된 모델 가중치 저장 (save_pretrained)
print(f"\n[3/3] Saving quantized model weights to: {SAVE_DIR} ...")
model.save_pretrained(SAVE_DIR)

print(f"\nSuccessfully quantized and saved model to '{SAVE_DIR}'!")