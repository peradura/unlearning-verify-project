import json
import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_pure_kl_divergence(model_orig, model_unlearn, tokenizer, text_input):
    """[Black-box 1] 두 모델 간의 출력 확률 분포 차이(KL-Divergence) 연산 함수"""
    model_orig.eval()
    model_unlearn.eval()

    device = next(model_orig.parameters()).device
    inputs = tokenizer(text_input, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs_orig = model_orig(**inputs)
        outputs_unlearn = model_unlearn(**inputs)

    logits_orig = outputs_orig.logits.float()
    logits_unlearn = outputs_unlearn.logits.float()

    prob_orig = F.softmax(logits_orig, dim=-1)
    log_prob_unlearn = F.log_softmax(logits_unlearn, dim=-1)

    kl_loss = F.kl_div(log_prob_unlearn, prob_orig, reduction='batchmean')
    return kl_loss.item()


if __name__ == "__main__":
    # 1. 고정된 정밀 모델 수납 보관소 경로 세팅
    CONFIG_TOKENIZER_PATH = "../upload/model/config/llama_1b"
    ORIG_PTH_PATH = "../upload/model/pth/original/llama_1b_org"
    UNLEARN_PATH = "../upload/model/pth/unlearned/llama_1b_unlearned"

    print("🤖 12조 대조 검증을 위해 모델 및 토크나이저 세트 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG_TOKENIZER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    m_orig = AutoModelForCausalLM.from_pretrained(ORIG_PTH_PATH, config=CONFIG_TOKENIZER_PATH,
                                                  torch_dtype=torch.float16, device_map="auto")
    m_unlearn = AutoModelForCausalLM.from_pretrained(UNLEARN_PATH, config=CONFIG_TOKENIZER_PATH,
                                                     torch_dtype=torch.float16, device_map="auto")

    # --------------------------------------------------------
    # 🎯 [핵심 자동화] 업로드된 데이터셋을 그대로 불러와 전수 검사
    # --------------------------------------------------------
    UPLOADED_DATASET_PATH = "./upload/unlearned_dataset/dataset.json"  # 사용자가 올린 실제 데이터셋 경로

    with open(UPLOADED_DATASET_PATH, "r", encoding="utf-8") as f:
        uploaded_delete_set = json.load(f)

    print(f"\n📈 파일 로드 완료! 총 {len(uploaded_delete_set)}개의 삭제 대상 데이터 전수 검증을 시작합니다.")

    kl_scores = []

    # 데이터셋에 들어있는 개수 그대로(10개든 100개든) 루프를 돌며 전부 계산합니다.
    for idx, sample in enumerate(uploaded_delete_set, 1):
        # 1. 프롬프트 형식을 그대로 1차원 평서문으로 결합
        combined_text = f"{sample['prompt'].strip()} {sample['response'].strip()}"

        # 2. 그대로 KL 함수에 주입
        score = compute_pure_kl_divergence(m_orig, m_unlearn, tokenizer, combined_text)
        kl_scores.append(score)

        print(f" └ [진행 {idx}/{len(uploaded_delete_set)}] 샘플 KL 점수: {score:.6f}")

    # 3. 데이터셋 전체 샘플의 최종 평균 KL 산출
    final_avg_kl = sum(kl_scores) / len(kl_scores)

    print("\n========================================================")
    print(f"📊 [최종 결과] 해당 데이터셋 전체 평균 KL-Divergence: {final_avg_kl:.6f}")
    print("========================================================")