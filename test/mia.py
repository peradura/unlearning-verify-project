import json
import os
import math
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score, accuracy_score

# 🧠 모델 레지스트리 (CPU 상주용)
MODEL_REGISTRY = {
    "original": None,
    "unlearned": None
}


def load_model_cpu(source_folder, target_type="original"):
    if not os.path.exists(source_folder):
        print(f"❌ 경로 부재: {source_folder}")
        return False
    try:
        tokenizer = AutoTokenizer.from_pretrained(source_folder, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            source_folder,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        model.eval()
        MODEL_REGISTRY[target_type] = model
        return True
    except Exception as e:
        print(f"❌ 로드 실패: {e}")
        return False


def compute_loss(model, tokenizer, text_input):
    """MIA의 공격 피처(Feature)로 사용할 순수 크로스 엔트로피 Loss 계산"""
    inputs = tokenizer(text_input, return_tensors="pt").to("cpu")
    labels = inputs.input_ids.clone()
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
    return outputs.loss.item()


def extract_losses_from_json(model, tokenizer, file_path, max_samples=100):
    """데이터셋을 읽어 각 샘플의 Loss 배열을 반환"""
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    losses = []
    # 상위 max_samples개만 고속 추출
    for sample in dataset[:max_samples]:
        combined_text = f"{sample['prompt'].strip()} {sample['response'].strip()}"
        loss = compute_loss(model, tokenizer, combined_text)
        losses.append(loss)
    return losses


def evaluate_mia_performance(member_losses, non_member_losses):
    """
    Loss 값을 기반으로 MIA 공격 성능(AUC, Best Accuracy)을 계산합니다.
    일반적으로 학습 데이터(Member)는 Loss가 낮고, 미학습 데이터(Non-Member)는 Loss가 높습니다.
    """
    # 실제 정답 라벨 (Member = 1, Non-Member = 0)
    y_true = np.concatenate([np.ones(len(member_losses)), np.zeros(len(non_member_losses))])

    # 공격 모델은 Loss가 낮을수록 Member(1)일 확률이 높다고 판단하므로,
    # AUC 계산을 위해 Loss에 마이너스를 붙여 '멤버 점수(Score)'로 변환합니다.
    member_scores = -np.array(member_losses)
    non_member_scores = -np.array(non_member_losses)
    y_scores = np.concatenate([member_scores, non_member_scores])

    # 1. MIA 공격 AUC-ROC 스코어 계산
    mia_auc = roc_auc_score(y_true, y_scores)

    # 2. 최적의 임계치(Threshold)를 찾아서 공격 정확도(Best Accuracy) 계산
    best_acc = 0.0
    all_scores = sorted(y_scores)
    for thresh in all_scores:
        y_pred = (y_scores >= thresh).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc

    return mia_auc, best_acc


if __name__ == "__main__":
    PATH_ORIGINAL = "./model_check/test_orig"
    PATH_UNLEARNED = "./model_check/test_unlearn"

    print("🔄 모델 상주 프로세스 가동...")
    orig_ok = load_model_cpu(PATH_ORIGINAL, target_type="original")
    unlearn_ok = load_model_cpu(PATH_UNLEARNED, target_type="unlearned")

    if orig_ok and unlearn_ok:
        tokenizer = AutoTokenizer.from_pretrained(PATH_ORIGINAL, trust_remote_code=True)
        model_orig = MODEL_REGISTRY["original"]
        model_unlearn = MODEL_REGISTRY["unlearned"]

        # 📂 MIA 평가를 위한 데이터셋 경로 (Holdout은 완벽한 미학습 상식 데이터셋 가정)
        FORGET_DATA_PATH = "../dataset/forget10_data.json"  # ❌ 공격 타겟 (Member로 가정)
        RETAIN_DATA_PATH = "../dataset/retain90_data.json"  # 🛡️ 대조군 (Non-Member 대용 혹은 별도 Holdout 파일)

        print("\n" + "=" * 60)
        print("🕵️‍♂️ [MIA 공격 시뮬레이션] 각 모델별 가중치 특징 추출 중... (각 100개 샘플)")
        print("=" * 60)

        # 1. 원본 모델의 Loss 특징 추출
        print("[1/4] 원본 모델 -> Forget 데이터 Loss 추출 중...")
        orig_member_losses = extract_losses_from_json(model_orig, tokenizer, FORGET_DATA_PATH, max_samples=100)
        print("[2/4] 원본 모델 -> Retain 데이터 Loss 추출 중...")
        orig_non_member_losses = extract_losses_from_json(model_orig, tokenizer, RETAIN_DATA_PATH, max_samples=100)

        # 2. 언러닝 모델의 Loss 특징 추출
        print("[3/4] 언러닝 모델 -> Forget 데이터 Loss 추출 중...")
        unlearn_member_losses = extract_losses_from_json(model_unlearn, tokenizer, FORGET_DATA_PATH, max_samples=100)
        print("[4/4] 언러닝 모델 -> Retain 데이터 Loss 추출 중...")
        unlearn_non_member_losses = extract_losses_from_json(model_unlearn, tokenizer, RETAIN_DATA_PATH,
                                                             max_samples=100)

        # 3. MIA 공격 지표 연산
        orig_auc, orig_acc = evaluate_mia_performance(orig_member_losses, orig_non_member_losses)
        unlearn_auc, unlearn_acc = evaluate_mia_performance(unlearn_member_losses, unlearn_non_member_losses)

        # --------------------------------------------------------
        # 🏁 MIA 최종 방어 리포트 출력
        # --------------------------------------------------------
        print("\n" + "=" * 60)
        print("🏁 [MIA(멤버십 추론 공격) 최종 검증 보고서]")
        print("=" * 60)
        print(f"📊 원본 모델 (Original Model) 공격 결과")
        print(f"  ├ 공격 성공률 (AUC-ROC) : {orig_auc:.4f}")
        print(f"  └ 공격 최적 정확도 (Acc) : {orig_acc * 100:.2f}%")
        print(f"  💡 해석: 공격자가 해당 지식이 학습에 쓰였는지 아주 잘 맞추는 상태 (기억 선명)")
        print("-" * 60)
        print(f"📊 언러닝 모델 (Unlearned Model) 공격 결과")
        print(f"  ├ 공격 성공률 (AUC-ROC) : {unlearn_auc:.4f} 🎯")
        print(f"  └ 공격 최적 정확도 (Acc) : {unlearn_acc * 100:.2f}%")
        print(f"  💡 해석: 수치가 0.50(50%)에 가까워질수록 완벽하게 지식을 은닉/망각했다는 뜻")
        print("=" * 60)

        # 학술적 Delta 리포트
        print(f"📈 MIA 공격 차단력 변화 (AUC Delta): {orig_auc - unlearn_auc:.4f} 감소 수납 완료")

    else:
        print("❌ 모델 인프라 경로를 확인해 주세요.")