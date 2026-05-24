import json
import os
import re
import math
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# 🧠 모델 레지스트리 (CPU 상주용)
MODEL_REGISTRY = {
    "original": None,
    "unlearned": None
}


def load_and_verify_model_pipeline_cpu(source_folder, target_type="original"):
    print("\n" + "=" * 50)
    print(f"🖥️ [CPU 모드] 모델 검증 및 RAM 상주 시작 [{target_type.upper()}]")
    print(f"📂 경로: {source_folder}")
    print("=" * 50)

    if not os.path.exists(source_folder):
        print(f"❌ [오류] 지정된 모델 폴더 경로가 존재하지 않습니다: {source_folder}")
        return False

    try:
        config = AutoConfig.from_pretrained(source_folder)
        torch_dtype = torch.float16
        tokenizer = AutoTokenizer.from_pretrained(source_folder, trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(
            source_folder,
            torch_dtype=torch_dtype,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        model.eval()
        MODEL_REGISTRY[target_type] = model
        print(f"✅ [성공] {target_type} 모델 로드 완료!")
        return True
    except Exception as e:
        print(f"❌ 실패: {e}")
        return False


def run_layer_l2_experiment():
    """[화이트박스] 가중치 수치 실측 (L2)"""
    model_orig = MODEL_REGISTRY["original"]
    model_unlearn = MODEL_REGISTRY["unlearned"]

    print("\n" + "=" * 50)
    print("🔬 [실험 1] 레이어별 가중치 행렬 L2 Distance 실측 결과")
    print("=" * 50)

    layers_orig = model_orig.model.layers
    layers_unlearn = model_unlearn.model.layers

    for i in range(len(layers_orig)):
        w_orig = layers_orig[i].self_attn.q_proj.weight.data
        w_unlearn = layers_unlearn[i].self_attn.q_proj.weight.data
        l2_dist = torch.norm(w_orig - w_unlearn, p=2).item()
        print(f"   └ [Layer {i:02d}] Attention Q-Projection L2 Distance: {l2_dist:.6f}")


def compute_model_metrics(model, tokenizer, text_input):
    """[블랙박스 핵심] 특정 문장에 대해 정답 로그 확률(오픈언러닝 규격)과 Perplexity를 동시 계산"""
    inputs = tokenizer(text_input, return_tensors="pt").to("cpu")
    labels = inputs.input_ids.clone()

    with torch.no_grad():
        outputs = model(**inputs, labels=labels)

    loss = outputs.loss.item()
    log_likelihood = -loss

    try:
        perplexity = math.exp(loss)
    except OverflowError:
        perplexity = float('inf')

    return log_likelihood, perplexity


def evaluate_dataset_metrics(model_orig, model_unlearn, tokenizer, dataset_path, dataset_name="Target",
                             max_samples=100):
    """JSON 파일을 읽어서 지정된 샘플 수(기본 100개)만큼만 제한하여 지표 도출"""
    if not os.path.exists(dataset_path):
        print(f"⚠️ [{dataset_name}] 데이터 파일이 없습니다: {dataset_path} (스킵)")
        return None

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 🎯 사용자가 요청한 개수(100개)로 데이터 서브셋 슬라이싱
    target_subset = dataset[:max_samples]
    total_len = len(target_subset)

    print(f"\n📈 [{dataset_name}] 총 {total_len}개 샘플 블랙박스 지표 전수 스캔... (최대 {max_samples}개 제한 적용)")

    orig_log_list, unlearn_log_list = [], []
    orig_ppl_list, unlearn_ppl_list = [], []

    for idx, sample in enumerate(target_subset, 1):
        combined_text = f"{sample['prompt'].strip()} {sample['response'].strip()}"

        # 원본 모델 측정
        orig_log, orig_ppl = compute_model_metrics(model_orig, tokenizer, combined_text)
        orig_log_list.append(orig_log)
        orig_ppl_list.append(orig_ppl)

        # 언러닝 모델 측정
        unlearn_log, unlearn_ppl = compute_model_metrics(model_unlearn, tokenizer, combined_text)
        unlearn_log_list.append(unlearn_log)
        unlearn_ppl_list.append(unlearn_ppl)

        # 100개 검증이므로 10개 단위로 촘촘하게 진척도 확인
        if idx % 10 == 0 or idx == total_len:
            print(f" └ [{dataset_name} 진행 {idx}/{total_len}] "
                  f"현재 평균 -> 원본 PPL: {sum(orig_ppl_list) / len(orig_ppl_list):.2f} | "
                  f"언러닝 PPL: {sum(unlearn_ppl_list) / len(unlearn_ppl_list):.2f}")

    return {
        "orig_log_avg": sum(orig_log_list) / len(orig_log_list),
        "unlearn_log_avg": sum(unlearn_log_list) / len(unlearn_log_list),
        "orig_ppl_avg": sum(orig_ppl_list) / len(orig_ppl_list),
        "unlearn_ppl_avg": sum(unlearn_ppl_list) / len(unlearn_ppl_list)
    }


if __name__ == "__main__":
    PATH_ORIGINAL = "./model_check/test_orig"
    PATH_UNLEARNED = "./model_check/test_unlearn"

    orig_ok = load_and_verify_model_pipeline_cpu(PATH_ORIGINAL, target_type="original")
    unlearn_ok = load_and_verify_model_pipeline_cpu(PATH_UNLEARNED, target_type="unlearned")

    if orig_ok and unlearn_ok:
        # [실험 1] 화이트박스 실측
        run_layer_l2_experiment()

        tokenizer = AutoTokenizer.from_pretrained(PATH_ORIGINAL, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_orig = MODEL_REGISTRY["original"]
        model_unlearn = MODEL_REGISTRY["unlearned"]

        # [실험 2] TOFU 10% vs 90% 블랙박스 지표 고속 검증 (각 100개 스케일)
        print("\n" + "=" * 60)
        print("📊 [실험 2] 오픈 언러닝 공식 로그 확률 & Perplexity 고속 검증 개시 (100개 샘플)")
        print("=" * 60)

        FORGET_DATA_PATH = "../dataset/forget10_data.json"  # ❌ Forget 10%
        RETAIN_DATA_PATH = "../dataset/retain90_data.json"  # 🛡️ Retain 90%

        # max_samples="100" 인자를 주입하여 스캔 범위 제어
        forget_res = evaluate_dataset_metrics(model_orig, model_unlearn, tokenizer, FORGET_DATA_PATH, "Forget-10%",
                                              max_samples=100)
        retain_res = evaluate_dataset_metrics(model_orig, model_unlearn, tokenizer, RETAIN_DATA_PATH, "Retain-90%",
                                              max_samples=100)

        # --------------------------------------------------------
        # 🏁 최종 리포트 출력 파트
        # --------------------------------------------------------
        print("\n" + "=" * 60)
        print("🏁 [최종 공식 대조 보고서 - 100개 샘플 요약]")
        print("=" * 60)

        if forget_res:
            print("[❌ Forget-10% 세트 (지식 삭제 타겟)]")
            print(
                f"  ├ 원본 모델   -> 로그 확률: {forget_res['orig_log_avg']:.4f} | Perplexity: {forget_res['orig_ppl_avg']:.2f}")
            print(
                f"  └ 언러닝 모델 -> 로그 확률: {forget_res['unlearn_log_avg']:.4f} | Perplexity: {forget_res['unlearn_ppl_avg']:.2f} 🔥")

        if retain_res:
            print("\n[🛡️ Retain-90% 세트 (일반 상식 보존 대조군)]")
            print(
                f"  ├ 원본 모델   -> 로그 확률: {retain_res['orig_log_avg']:.4f} | Perplexity: {retain_res['orig_ppl_avg']:.2f}")
            print(
                f"  └ 언러닝 모델 -> 로그 확률: {retain_res['unlearn_log_avg']:.4f} | Perplexity: {retain_res['unlearn_ppl_avg']:.2f} 🛡️")
        print("=" * 60)

    else:
        print("\n❌ 모델 경로 인프라를 확인해 주세요.")