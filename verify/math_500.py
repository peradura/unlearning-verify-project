import json
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset  # 🌟 원격 스트리밍을 위한 라이브러리


def evaluate_math_500_sample(model, tokenizer, sample, max_new_tokens=512):
    """
    [MATH-500 전용 주관식 평가]
    허깅페이스 스트리밍 데이터 구조(problem, answer)를 파싱하여 CoT 추론 후 대조합니다.
    """
    model.eval()
    device = next(model.parameters()).device

    # 허깅페이스 공식 MATH-500은 질문 필드명이 'instruction'이 아니라 'problem'입니다.
    problem = sample['problem']
    prompt = (
        f"Problem: {problem}\n"
        f"Let's think step by step and solve the problem. "
        f"Finally, provide your distinct answer inside \\boxed{{...}} (e.g., \\boxed{{42}}).\n"
        f"Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generation_output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.95,
            do_sample=False,  # Greedy Decoding 고정
            pad_token_id=tokenizer.eos_token_id
        )

    # 모델이 생성한 순수 답변 텍스트만 디코딩
    generated_text = tokenizer.decode(
        generation_output[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    # [정답 추출 로직]
    # 1순위: \boxed{정답} 추출
    boxed_match = re.findall(r"\\boxed\{([^}]+)\}", generated_text)
    if boxed_match:
        pred_answer = boxed_match[-1].strip()
    else:
        # 2순위: 실패 시 맨 마지막 숫자 추출
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", generated_text)
        pred_answer = numbers[-1].strip() if numbers else ""

    # 허깅페이스 원격 데이터셋의 정답 추출 및 비교
    true_answer = str(sample['answer']).strip()

    return 1 if pred_answer == true_answer else 0


def run_math500_streaming_experiment(m_orig, m_unlearn, tokenizer, limit_samples=10):
    """
    [원격 스트리밍] 로컬 파일 없이 허깅페이스에서 즉석으로 긁어와 주관식 수학 추론 검증 (N개 제한)
    """
    print("\n" + "=" * 65)
    print(f"🧮 [실험 3] MATH-500 실시간 스트리밍 대조 검증 (상위 {limit_samples}개 스크리닝)")
    print("=========================================================")

    try:
        # 🌟 공식 MATH-500 레포지토리에서 test 스플릿을 실시간 스트리밍으로 연결
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test", streaming=True)
    except Exception as e:
        print(f"   ❌ [MATH-500] 데이터 원격 스트리밍 로드 실패: {str(e)}")
        print("=========================================================")
        return {}

    orig_correct = 0
    unlearn_correct = 0
    count = 0

    print(f"   🚀 허깅페이스 서버로부터 데이터를 스트리밍하며 대조 연산을 수행합니다...\n")

    for sample in dataset:
        if count >= limit_samples:
            break

        count += 1

        # 실시간 각 문제 풀이 수행
        orig_res = evaluate_math_500_sample(m_orig, tokenizer, sample)
        unlearn_res = evaluate_math_500_sample(m_unlearn, tokenizer, sample)

        orig_correct += orig_res
        unlearn_correct += unlearn_res

        # 1B 모델의 주관식 풀이 속도를 고려하여 1문항 풀 때마다 실시간 진척도 로그 프린트
        current_orig_acc = (orig_correct / count) * 100
        current_unlearn_acc = (unlearn_correct / count) * 100
        print(
            f"   ⚡ [진행 {count:>2}/{limit_samples}] 실시간 정답률 -> 🔴 Orig: {current_orig_acc:>5.1f}% | 🔵 Unlearned: {current_unlearn_acc:>5.1f}%")

    print("-" * 65)

    # 최종 결과 출력
    if count > 0:
        global_orig_acc = (orig_correct / count) * 100
        global_unlearn_acc = (unlearn_correct / count) * 100

        print(f"   📊 [MATH-500 STREAMING FINAL ACCURACY] 총 검증 문항: {count}개")
        print(f"      🔴 Original 전체 최종 정답률 : {global_orig_acc:.2f}%")
        print(f"      🔵 Unlearned 전체 최종 정답률: {global_unlearn_acc:.2f}%")
    else:
        print("   ❌ 스트리밍된 데이터가 존재하지 않습니다.")

    print("=========================================================")

    return {
        "original_acc": global_orig_acc if count > 0 else 0,
        "unlearned_acc": global_unlearn_acc if count > 0 else 0,
        "total": count
    }


if __name__ == "__main__":
    # Rate Limit 우회를 위해 토크나이저 및 가중치 베이스는 안전하게 로컬 폴더 지정
    CONFIG_TOKENIZER_PATH = "../models/before"
    ORIG_PTH_PATH = "../models/before"
    UNLEARN_PATH = "../models/after"

    print("🤖 대조 검증을 위해 모델 및 토크나이저 세트 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG_TOKENIZER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # float16 로드
    m_orig = AutoModelForCausalLM.from_pretrained(ORIG_PTH_PATH, config=CONFIG_TOKENIZER_PATH,
                                                  torch_dtype=torch.float16, device_map="auto")
    m_unlearn = AutoModelForCausalLM.from_pretrained(UNLEARN_PATH, config=CONFIG_TOKENIZER_PATH,
                                                     torch_dtype=torch.float16, device_map="auto")

    # 🌟 우선 가볍고 빠르게 작동하는지 확인하기 위해 10개만 타겟으로 스트리밍 테스트 진행
    # 전체를 다 뽑고 싶다면 이 숫자를 500으로 늘리시면 됩니다.
    run_math500_streaming_experiment(m_orig, m_unlearn, tokenizer, limit_samples=500)