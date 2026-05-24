import json
import os
import re
import math
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score, accuracy_score

# 🧠 3대 자산 모델 레지스트리 (CPU 상주용)
MODEL_REGISTRY = {
    "original": None,
    "unlearned": None,
    "baseline": None
}

# 🪝 전 구간 뉴런 활성 강도를 누적할 수납 창고
act_store = {
    "original": {"total_val": 0.0, "active_count": 0, "total_count": 0, "vectors": []},
    "unlearned": {"total_val": 0.0, "active_count": 0, "total_count": 0, "vectors": []},
    "baseline": {"total_val": 0.0, "active_count": 0, "total_count": 0, "vectors": []}
}


def make_multi_layer_hook(model_type, thresh=0.01):
    """지정된 구간 내 모든 레이어에서 활성도를 가로채는 훅"""

    def hook_fn(module, input, output):
        act_val = output.detach().float().mean(dim=(0, 1)).abs()
        act_store[model_type]["total_val"] += act_val.sum().item()
        act_store[model_type]["active_count"] += (act_val > thresh).sum().item()
        act_store[model_type]["total_count"] += act_val.numel()
        act_store[model_type]["vectors"].append(act_val)

    return hook_fn


def load_and_verify_model_pipeline_cpu(source_folder, target_type="original"):
    print("\n" + "=" * 50)
    print(f"🖥️ [CPU 모드] 모델 로딩 및 RAM 상주 가동 [{target_type.upper()}]")
    print(f"📂 경로: {source_folder}")
    print("=" * 50)

    if not os.path.exists(source_folder):
        print(f"❌ [오류] 경로가 존재하지 않습니다: {source_folder}")
        return False

    try:
        # [최신 규격 보정] torch_dtype 경고를 완전히 제거하는 로딩 파트
        config = AutoConfig.from_pretrained(source_folder)
        model = AutoModelForCausalLM.from_pretrained(
            source_folder,
            dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        model.eval()
        MODEL_REGISTRY[target_type] = model
        print(f"✅ [성공] {target_type} 모델 메인 RAM 상주 완료!")
        return True
    except Exception as e:
        print(f"❌ 실패: {e}")
        return False


def compute_model_metrics(model, tokenizer, text_input):
    """[블랙박스 엔진] 로그 확률과 Perplexity 동시 연산"""
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

    return log_likelihood, perplexity, loss


def evaluate_mia_performance(member_losses, non_member_losses):
    """Loss 기반 임계치 MIA 공격 성공률 산출"""
    y_true = np.concatenate([np.ones(len(member_losses)), np.zeros(len(non_member_losses))])
    y_scores = np.concatenate([-np.array(member_losses), -np.array(non_member_losses)])

    mia_auc = roc_auc_score(y_true, y_scores)
    best_acc = 0.0
    for thresh in sorted(y_scores):
        y_pred = (y_scores >= thresh).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
    return mia_auc, best_acc


if __name__ == "__main__":
    # 📂 3대 자산 경로 설정 완료
    PATH_ORIGINAL = "./model_check/test_orig"
    PATH_UNLEARNED = "./model_check/test_unlearn"
    PATH_BASELINE = "./model_check/test_baseline"  # ◀️ 새로 수납한 미학습 대조군 추가!

    # 1. 3대 자산 로딩 및 검증 개시
    orig_ok = load_and_verify_model_pipeline_cpu(PATH_ORIGINAL, target_type="original")
    unlearn_ok = load_and_verify_model_pipeline_cpu(PATH_UNLEARNED, target_type="unlearned")
    baseline_ok = load_and_verify_model_pipeline_cpu(PATH_BASELINE, target_type="baseline")

    if orig_ok and unlearn_ok and baseline_ok:
        tokenizer = AutoTokenizer.from_pretrained(PATH_ORIGINAL, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_orig = MODEL_REGISTRY["original"]
        model_unlearn = MODEL_REGISTRY["unlearned"]
        model_base = MODEL_REGISTRY["baseline"]

        # --------------------------------------------------------
        # 🎯 아키텍처 자동 적응형 다중 레이어 훅 등록 (50% ~ 90% 구간)
        # --------------------------------------------------------
        total_layers = len(model_orig.model.layers)
        start_idx, end_idx = int(total_layers * 0.50), int(total_layers * 0.90)
        target_indices = list(range(start_idx, end_idx + 1))

        hooks = []
        for idx in target_indices:
            hooks.append(
                model_orig.model.layers[idx].mlp.gate_proj.register_forward_hook(make_multi_layer_hook("original")))
            hooks.append(
                model_unlearn.model.layers[idx].mlp.gate_proj.register_forward_hook(make_multi_layer_hook("unlearned")))
            hooks.append(
                model_base.model.layers[idx].mlp.gate_proj.register_forward_hook(make_multi_layer_hook("baseline")))

        # 📂 TOFU 데이터셋 로딩 경로 (유저님 기존 가이드 매핑)
        FORGET_DATA_PATH = "../dataset/forget10_data.json"
        RETAIN_DATA_PATH = "../dataset/retain90_data.json"

        with open(FORGET_DATA_PATH, "r", encoding="utf-8") as f:
            forget_db = json.load(f)
        with open(RETAIN_DATA_PATH, "r", encoding="utf-8") as f:
            retain_db = json.load(f)

        # 📊 통계 수집 컨테이너 세팅
        stats = {m: {"f_log": [], "f_ppl": [], "f_loss": [], "r_log": [], "r_ppl": [], "r_loss": []} for m in
                 MODEL_REGISTRY.keys()}

        print("\n" + "=" * 60)
        print("🚀 [실험 가동] 3대 모델 4대 지표 통합 스캔 개시 (각 100개 샘플 정밀 제한)")
        print("=" * 60)

        # [A] Forget 데이터 스캔
        print("📦 Forget-10% 세트 연산 중...")
        for sample in forget_db[:100]:
            text = f"{sample['prompt'].strip()} {sample['response'].strip()}"

            orig_log, orig_ppl, orig_loss = compute_model_metrics(model_orig, tokenizer, text)
            unlearn_log, unlearn_ppl, unlearn_loss = compute_model_metrics(model_unlearn, tokenizer, text)
            base_log, base_ppl, base_loss = compute_model_metrics(model_base, tokenizer, text)

            stats["original"]["f_log"].append(orig_log);
            stats["original"]["f_ppl"].append(orig_ppl);
            stats["original"]["f_loss"].append(orig_loss)
            stats["unlearned"]["f_log"].append(unlearn_log);
            stats["unlearned"]["f_ppl"].append(unlearn_ppl);
            stats["unlearned"]["f_loss"].append(unlearn_loss)
            stats["baseline"]["f_log"].append(base_log);
            stats["baseline"]["f_ppl"].append(base_ppl);
            stats["baseline"]["f_loss"].append(base_loss)

        # [B] Retain 데이터 스캔 (이때 등록된 훅이 뉴런 값을 캡처함)
        print("📦 Retain-90% 세트 연산 및 화이트박스 뇌지도 캡처 중...")
        for sample in retain_db[:100]:
            text = f"{sample['prompt'].strip()} {sample['response'].strip()}"

            orig_log, orig_ppl, orig_loss = compute_model_metrics(model_orig, tokenizer, text)
            unlearn_log, unlearn_ppl, unlearn_loss = compute_model_metrics(model_unlearn, tokenizer, text)
            base_log, base_ppl, base_loss = compute_model_metrics(model_base, tokenizer, text)

            stats["original"]["r_log"].append(orig_log);
            stats["original"]["r_ppl"].append(orig_ppl);
            stats["original"]["r_loss"].append(orig_loss)
            stats["unlearned"]["r_log"].append(unlearn_log);
            stats["unlearned"]["r_ppl"].append(unlearn_ppl);
            stats["unlearned"]["r_loss"].append(unlearn_loss)
            stats["baseline"]["r_log"].append(base_log);
            stats["baseline"]["r_ppl"].append(base_ppl);
            stats["baseline"]["r_loss"].append(base_loss)

        # 안전하게 전방 훅 해제
        for hk in hooks: hk.remove()

        # 🕵️‍♂️ MIA 공격 지표 산출
        orig_mia_auc, orig_mia_acc = evaluate_mia_performance(stats["original"]["f_loss"], stats["original"]["r_loss"])
        unlearn_mia_auc, unlearn_mia_acc = evaluate_mia_performance(stats["unlearned"]["f_loss"],
                                                                    stats["unlearned"]["r_loss"])
        base_mia_auc, base_mia_acc = evaluate_mia_performance(stats["baseline"]["f_loss"], stats["baseline"]["r_loss"])

        # 🧠 뉴런 활성 패턴 글로벌 코사인 유사도 추출
        v_orig = torch.stack(act_store["original"]["vectors"]).mean(dim=0)
        v_unlearn = torch.stack(act_store["unlearned"]["vectors"]).mean(dim=0)
        v_base = torch.stack(act_store["baseline"]["vectors"]).mean(dim=0)

        sim_orig_vs_unlearn = torch.nn.functional.cosine_similarity(v_orig, v_unlearn, dim=0).item()
        sim_orig_vs_baseline = torch.nn.functional.cosine_similarity(v_orig, v_base, dim=0).item()

        # --------------------------------------------------------
        # 🏁 대망의 3원 교차 실측 그랜드 마스터 리포트
        # --------------------------------------------------------
        print("\n" + "============================================================")
        print("🏁 [3원 교차 실측 그랜드 마스터 리포트 - 100개 샘플 완공]")
        print("============================================================")
        print(f"📈 가중치 스캔 대역폭 : 전체 {total_layers}개 레이어 중 {start_idx}번 ~ {end_idx}번 레이어")
        print("-" * 60)

        # 1. 원본 모델 리포트
        print("[📊 1. 원본 오리지널 모델 (Original)]")
        print(
            f"  ├ Forget 지식 억제력 ➔ 로그 확률: {np.mean(stats['original']['f_log']):.4f} | Perplexity: {np.mean(stats['original']['f_ppl']):.2f}")
        print(
            f"  ├ Retain 일반 상식   ➔ 로그 확률: {np.mean(stats['original']['r_log']):.4f} | Perplexity: {np.mean(stats['original']['r_ppl']):.2f}")
        print(f"  └ 프라이버시 방어    ➔ MIA AUC: {orig_mia_auc:.4f} | 공격 최적 정확도: {orig_mia_acc * 100:.1f}%")
        print("-" * 60)

        # 2. 언러닝 가중치 모델 리포트
        print("[🧠 2. 본 연구의 언러닝 가중치 모델 (Unlearned)]")
        print(
            f"  ├ Forget 지식 억제력 ➔ 로그 확률: {np.mean(stats['unlearned']['f_log']):.4f} | Perplexity: {np.mean(stats['unlearned']['f_ppl']):.2f} 🔥")
        print(
            f"  ├ Retain 일반 상식   ➔ 로그 확률: {np.mean(stats['unlearned']['r_log']):.4f} | Perplexity: {np.mean(stats['unlearned']['r_ppl']):.2f} 🛡️")
        print(f"  ├ 프라이버시 방어    ➔ MIA AUC: {unlearn_mia_auc:.4f} | 공격 최적 정확도: {unlearn_mia_acc * 100:.1f}%")
        print(f"  └ 뇌 지도 계승 밀도  ➔ 원본 모델 대비 구간 구조 유사도: {sim_orig_vs_unlearn:.4f} ✨")
        print("-" * 60)

        # 3. 신규 도입 Baseline 모델 리포트
        print("[🍏 3. 처음부터 안 배운 공식 대조군 모델 (Baseline/Retain-Only)]")
        print(
            f"  ├ Forget 지식 억제력 ➔ 로그 확률: {np.mean(stats['baseline']['f_log']):.4f} | Perplexity: {np.mean(stats['baseline']['f_ppl']):.2f} 🟢")
        print(
            f"  ├ Retain 일반 상식   ➔ 로그 확률: {np.mean(stats['baseline']['r_log']):.4f} | Perplexity: {np.mean(stats['baseline']['r_ppl']):.2f}")
        print(f"  ├ 프라이버시 방어    ➔ MIA AUC: {base_mia_auc:.4f} | 공격 최적 정확도: {base_mia_acc * 100:.1f}%")
        print(f"  └ 뇌 지도 계승 밀도  ➔ 원본 모델 대비 구간 구조 유사도: {sim_orig_vs_baseline:.4f} ⚠️")
        print("============================================================")
        print("🎉 3대 자산 4차원 물리/확률 교차 검증 파이프라인 수납이 정상 종료되었습니다.")
        print("============================================================")

    else:
        print("❌ 3개 모델 자산 중 일부를 RAM에 바인딩하지 못했습니다. 로컬 수납 상태를 확인해 주세요.")