import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Pure Response Log-Probability Evaluator")
    parser.add_argument("--orig_path", type=str, required=True, help="Original model path")
    parser.add_argument("--unlearn_path", type=str, required=True, help="Unlearned model path")
    parser.add_argument("--forget_data", type=str, required=True, help="Forget dataset path (.json)")
    parser.add_argument("--max_samples", type=int, default=100, help="평가할 최대 샘플 수")
    return parser.parse_args()


def load_model_only(model_path):
    print(f"🔄 Loading Model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model.eval()
    return model


def compute_pure_response_log_prob(model, tokenizer, prompt, response):
    prompt_text = prompt.strip()
    full_text = f"{prompt_text} {response.strip()}"

    device = model.device
    inputs_full = tokenizer(full_text, return_tensors="pt").to(device)
    full_ids = inputs_full.input_ids

    inputs_prompt = tokenizer(prompt_text, return_tensors="pt")
    prompt_len = inputs_prompt.input_ids.shape[1]
    full_len = full_ids.shape[1]

    if full_len <= prompt_len:
        return None

    with torch.no_grad():
        outputs = model(**inputs_full)
        logits = outputs.logits.float()

    log_probs = torch.log_softmax(logits, dim=-1)
    shift_log_probs = log_probs[0, :-1, :]
    shift_labels = full_ids[0, 1:]

    gathered_log_probs = shift_log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
    response_log_probs = gathered_log_probs[prompt_len - 1:]

    return response_log_probs.mean().item()


def run_log_prob_evaluation(orig_path, unlearn_path, forget_data_path, max_samples=100):
    """
    🎯 [변수 리턴형 핵심 메인 함수]
    파일 저장을 수행하지 않고, 최종 결과 딕셔너리를 메모리 상의 변수로 리턴합니다.
    """
    # 1. 토크나이저 및 모델 세트 로드
    tokenizer = AutoTokenizer.from_pretrained(orig_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    m_orig = load_model_only(orig_path)
    m_unlearn = load_model_only(unlearn_path)

    # 2. 데이터셋 로드
    with open(forget_data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"\n📊 총 {len(dataset)}개 중 최대 {max_samples}개 샘플 정답 레이블 연산 가동...")

    orig_scores = []
    unlearn_scores = []

    # 3. 전수 루프 계측
    for sample in tqdm(dataset[:max_samples], desc="Evaluating Pure Log Probs"):
        prompt = sample["prompt"]
        response = sample["response"]

        lp_orig = compute_pure_response_log_prob(m_orig, tokenizer, prompt, response)
        lp_unlearn = compute_pure_response_log_prob(m_unlearn, tokenizer, prompt, response)

        if lp_orig is not None and lp_unlearn is not None:
            orig_scores.append(lp_orig)
            unlearn_scores.append(lp_unlearn)

    # 4. 스코어 통계 산출
    avg_orig = np.mean(orig_scores)
    avg_unlearn = np.mean(unlearn_scores)
    avg_gap = avg_unlearn - avg_orig

    # 5. 🎯 [결과 수납] 파일로 쓰지 않고, 변수에 할당할 딕셔너리 구조체 빌드
    evaluation_results = {
        "sample_size": len(orig_scores),
        "avg_original_log_prob": float(avg_orig),
        "avg_unlearned_log_prob": float(avg_unlearn),
        "pure_log_prob_gap": float(avg_gap)
    }

    # 🏁 터미널 온모니터링 리포트
    print("\n" + "========================================================")
    print("🏁 [순수 정답 구간] 오리지널 vs 언러닝 로그 확률 대조 성적표")
    print("========================================================")
    print(f" ├ ① Original 원본 모델 순수 정답 로그 확률   : {avg_orig:.6f}")
    print(f" ├ ② Unlearned 연구 모델 순수 정답 로그 확률 : {avg_unlearn:.6f}")
    print(f" └ 🔥 두 자산 간의 순수 확신도 이격 격차 (Gap)  : {avg_gap:.6f}")
    print("========================================================")

    # 메모리 자산 리턴
    return evaluation_results
