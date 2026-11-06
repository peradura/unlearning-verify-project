import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 🧠 2대 자산 모델 레지스트리 (CPU 상주용)
MODEL_REGISTRY = {
    "original": None,
    "unlearned": None
}


def load_and_verify_model_pipeline_cpu(source_folder, target_type="original"):
    print(f"🖥️ [CPU 모드] 모델 로딩 가동 [{target_type.upper()}] - 경로: {source_folder}")
    if not os.path.exists(source_folder):
        print(f"❌ [오류] 경로가 존재하지 않습니다: {source_folder}")
        return False
    try:
        model = AutoModelForCausalLM.from_pretrained(
            source_folder,
            dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        model.eval()
        MODEL_REGISTRY[target_type] = model
        print(f"✅ {target_type} 모델 로딩 완료!")
        return True
    except Exception as e:
        print(f"❌ 실패: {e}")
        return False


def generate_direct_output(model, tokenizer, prompt, max_new_tokens=50):
    """[직접 출력 비교용] 텍스트 생성 함수"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False  # 일관된 비교를 위해 Greedy Search 적용
        )
    # 프롬프트 부분은 제거하고 생성된 텍스트만 깔끔하게 반환
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()


if __name__ == "__main__":
    # 📂 2대 자산 경로 설정
    PATH_ORIGINAL = "../models/before"
    PATH_UNLEARNED = "../models/after"

    # 1. 자산 로딩 및 검증 개시
    orig_ok = load_and_verify_model_pipeline_cpu(PATH_ORIGINAL, target_type="original")
    unlearn_ok = load_and_verify_model_pipeline_cpu(PATH_UNLEARNED, target_type="unlearned")

    if orig_ok and unlearn_ok:
        tokenizer = AutoTokenizer.from_pretrained(PATH_ORIGINAL, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_orig = MODEL_REGISTRY["original"]
        model_unlearn = MODEL_REGISTRY["unlearned"]

        # 📂 Forget 데이터셋 로딩
        FORGET_DATA_PATH = "../dataset/forget10_data.json"

        with open(FORGET_DATA_PATH, "r", encoding="utf-8") as f:
            forget_db = json.load(f)

        print("\n" + "=" * 70)
        print("🚀 [실험 가동] Original vs Unlearned | Forget 프롬프트 텍스트 출력 비교")
        print("=" * 70)

        # 2. 텍스트 생성 및 직접 비교 (Top 10)
        # 요구하신 대로 정확히 10개만 순회합니다.
        for i, sample in enumerate(forget_db[:10]):
            prompt = sample['prompt'].strip()
            print(f"🔹 [Prompt {i + 1}/10] {prompt}")

            orig_output = generate_direct_output(model_orig, tokenizer, prompt)
            unlearn_output = generate_direct_output(model_unlearn, tokenizer, prompt)

            print(f"  🔴 [Original Output]  : {orig_output}")
            print(f"  🟢 [Unlearned Output] : {unlearn_output}")
            print("-" * 70)

        print("🎉 10개 샘플에 대한 텍스트 생성 직접 비교가 완료되었습니다.")

    else:
        print("❌ 모델 자산 중 일부를 로딩하지 못했습니다. 로컬 파일 및 경로를 다시 확인해 주세요.")