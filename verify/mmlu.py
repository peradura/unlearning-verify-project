import json
import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset  # 🌟 원격 스트리밍을 위한 라이브러리


def evaluate_mmlu_sample(model, tokenizer, sample, target_choices=['A', 'B', 'C', 'D']):
    model.eval()
    device = next(model.parameters()).device

    # 원격 데이터 구조를 1B 모델이 인식할 수 있는 객관식 포맷으로 가공
    choices_str = f"\nA) {sample['choices'][0]}\nB) {sample['choices'][1]}\nC) {sample['choices'][2]}\nD) {sample['choices'][3]}"
    prompt = f"Question: {sample['question']}{choices_str}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    next_token_logits = outputs.logits[0, -1, :]

    choice_logits = []
    for choice in target_choices:
        token_id = tokenizer.encode(choice, add_special_tokens=False)[-1]
        choice_logits.append(next_token_logits[token_id].item())

    pred_idx = choice_logits.index(max(choice_logits))
    pred_char = target_choices[pred_idx]

    # 원격 데이터셋의 answer인 정수(0~3)를 A~D 문자로 변환하여 대조
    mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    true_char = mapping[sample['answer']]

    return 1 if pred_char == true_char else 0


def run_mmlu_streaming_experiment(m_orig, m_unlearn, tokenizer, limit_per_subject=10):
    """
    [원격 스트리밍] 로컬 파일 없이 허깅페이스에서 즉석으로 긁어와 대조 검증 (과목당 N개 제한)
    """
    print("\n" + "=" * 65)
    print(f"🧠 [실험 2] MMLU 6종 실시간 스트리밍 대조 검증 (과목당 {limit_per_subject}개 스크리닝)")
    print("=========================================================")

    mmlu_subjects = [
        'global_facts',
        'abstract_algebra',
        'econometrics',
        'international_law',
        'marketing',
        'professional_accounting'
    ]

    summary_results = {}
    total_global_samples = 0
    total_orig_correct = 0
    total_unlearn_correct = 0

    for subject in mmlu_subjects:
        try:
            # 🌟 streaming=True 옵션으로 파일 다운로드 없이 실시간 데이터 스트림만 연결
            dataset = load_dataset("cais/mmlu", subject, split="test", streaming=True)
        except Exception as e:
            print(f"   ⚠️ [{subject:<23}] 데이터 원격 로드 실패: {str(e)}")
            continue

        orig_correct = 0
        unlearn_correct = 0
        count = 0

        # 스트리밍 데이터에서 설정한 개수(limit_per_subject)만큼만 뽑아서 연산
        for sample in dataset:
            if count >= limit_per_subject:
                break

            orig_correct += evaluate_mmlu_sample(m_orig, tokenizer, sample)
            unlearn_correct += evaluate_mmlu_sample(m_unlearn, tokenizer, sample)
            count += 1

        orig_accuracy = (orig_correct / count) * 100 if count > 0 else 0
        unlearn_accuracy = (unlearn_correct / count) * 100 if count > 0 else 0

        total_global_samples += count
        total_orig_correct += orig_correct
        total_unlearn_correct += unlearn_correct

        print(
            f"   └ [{subject:<23}] 검증문항: {count:>3}개 | 🔴 Orig: {orig_accuracy:>5.1f}% | 🔵 Unlearned: {unlearn_accuracy:>5.1f}%")

    print("-" * 65)

    if total_global_samples > 0:
        global_orig_acc = (total_orig_correct / total_global_samples) * 100
        global_unlearn_acc = (total_unlearn_correct / total_global_samples) * 100

        print(f"   📊 [TOTAL INTEGRATED ACCURACY] 총 검증 문항: {total_global_samples}개")
        print(f"      🔴 Original 전체 통합 정답률 : {global_orig_acc:.2f}%")
        print(f"      🔵 Unlearned 전체 통합 정답률: {global_unlearn_acc:.2f}%")
    else:
        print("   ❌ 스트리밍할 수 있는 데이터가 없습니다.")

    print("=========================================================")


if __name__ == "__main__":
    # 🌟 [수정] 토크나이저만큼은 검증된 허깅페이스 원격 경로에서 안전하게 가져옵니다.
    # 만약 사용 중인 모델이 Qwen이라면 "Qwen/Qwen2.5-1.5B-Instruct" 등으로 적어주세요.
    CONFIG_TOKENIZER_PATH = "../models/before"

    # 가중치 파일 로드는 기존 로컬 경로를 그대로 유지합니다.
    ORIG_PTH_PATH = "../models/before"
    UNLEARN_PATH = "../models/after"

    print("🤖 대조 검증을 위해 모델 및 토크나이저 세트 로드 중...")

    # 이제 가상환경 패키지 꼬임 없이 인터넷에서 깨끗한 설정을 원격으로 빌려옵니다.
    tokenizer = AutoTokenizer.from_pretrained(CONFIG_TOKENIZER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 가중치 로드는 그대로 로컬 폴더에서 정상 진행됩니다.
    m_orig = AutoModelForCausalLM.from_pretrained(ORIG_PTH_PATH, config=CONFIG_TOKENIZER_PATH,
                                                  torch_dtype=torch.float16, device_map="auto")
    m_unlearn = AutoModelForCausalLM.from_pretrained(UNLEARN_PATH, config=CONFIG_TOKENIZER_PATH,
                                                     torch_dtype=torch.float16, device_map="auto")

    # 실험 함수 실행
    run_mmlu_streaming_experiment(m_orig, m_unlearn, tokenizer, limit_per_subject=100)