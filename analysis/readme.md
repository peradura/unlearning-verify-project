# 🤖 TOFU Unlearning Evaluation Pipeline for Llama-3.2-1B-Instruct

이 프로젝트는 **Llama-3.2-1B-Instruct** 모델을 대상으로 **TOFU 데이터셋(`forget10`)** 기반의 3가지 머신 언러닝(Machine Unlearning) 기법(**GradDiff, AltPO, NPO**)을 수행하고 평가하기 위한 모듈식 파이프라인입니다.

언러닝 학습, 재학습 및 4-bit 양자화를 통해 총 11개 모델 변체(Base, Retain-only, FP16 언러닝 3종, 50% Forget 재학습 3종, 4-bit 양자화 3종)를 생성하고 7가지 표준 지표를 통해 대조·분석합니다.

---

## 📜 파일 구조 및 스크립트 설명

| 파일명 | 구분 | 설명 및 역할 |
| :--- | :--- | :--- |
| **`evaluator.py`** | **Core Library** | 7가지 핵심 평가 지표(MIA, $L_2$ Distance, Neuron Activation, PPL/LogProb, KL Divergence, MMLU 500샘플, GSM8K 200샘플)를 산출하는 모듈식 라이브러리 함수 모음입니다. |
| **`unlearning.py`** | **Train Pipeline** | GradDiff, AltPO, NPO 기법을 적용하여 언러닝 모델(FP16 및 50% Forget 재학습 모델)을 생성하고 학습하는 스크립트입니다. |
| **`quantization.py`** | **Model Export** | 학습된 언러닝 모델을 BitsAndBytes 등을 활용해 4-bit(NF4/FP4) 양자화 모델로 변환하여 출력·저장하는 스크립트입니다. |
| **`eval_standard.py`** | **Baseline Pipeline** | 기준 모델(`1_base`, `2_retain_only`) 2가지 전용 평가 스크립트입니다. 상대 지표를 제외한 모델 본연의 절대 지표(Lower/Upper Bound)를 추출하여 `./eval_standard_results/`에 저장합니다. |
| **`eval_unlearned.py`** | **Target Pipeline** | 언러닝 변체 모델 전용 평가 스크립트입니다. 지표를 추출한 후 `eval_standard.py`에서 얻은 기준 모델 2종의 결과와 비교하여 상대 수치(`_vs_base`, `_vs_retain`)를 포함해 종합 평가 결과를 `./eval_results/`에 저장합니다. |

---

## 📊 7가지 평가 지표 요약

1. **MIA (Membership Inference Attack)**: Forget 데이터셋에 대한 추론 성공률 (Loss Threshold 방식).
2. **$L_2$ Distance**: Base Model 대비 파라미터 가중치 변화량.
3. **Neuron Activation**: 모델 중간 레이어의 평균 뉴런 활성화 수치.
4. **Perplexity (PPL) & Log Likelihood**: Forget 데이터셋에 대한 모델의 암기 억제율.
5. **KL Divergence**: Retain 데이터셋에 대한 Base/Retain-only 모델 대비 출력 분포 유지력.
6. **MMLU Accuracy**: 일반 다중선택 지식 유지 성능 (500 샘플, Logprob 기반).
7. **GSM8K Accuracy**: 일반 수학 추론 능력 유지 성능 (200 샘플, Greedy Generation 기반).

---

## 🚀 파이프라인 실행 순서 (Workflow)

```bash
# 1. 기준 모델 2종(Base, Retain-only) 절대 지표 추출
python eval_standard.py

# 2. 언러닝 변체 모델 지표 추출 및 기준 모델 2종 대비 종합 평가
python eval_unlearned.py

# 3. 언러닝 모델 학습 및 재학습 모델 생성
python unlearning.py

# 4. 언러닝 모델 4-bit 양자화 적용
python quantization.py
