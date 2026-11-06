# 🤖 TOFU Unlearning Evaluation Pipeline for Llama-3.2-1B-Instruct

이 프로젝트는 **Llama-3.2-1B-Instruct** 모델을 대상으로 **TOFU 데이터셋(`forget10`)** 기반의 3가지 머신 언러닝(Machine Unlearning) 기법(**GradDiff, AltPO, NPO**)을 평가하기 위한 모듈식 파이프라인입니다. 

총 11개 모델 변체(Base, Retain-only, FP16 언러닝 3종, 50% Forget 재학습 3종, 4-bit 양자화 3종)의 성과를 7가지 표준 지표를 통해 대조·분석합니다.

---

## 📜 파일 구조 및 스크립트 설명

| 파일명 | 구분 | 설명 및 역할 |
| :--- | :--- | :--- |
| **`evaluator.py`** | **Core Library** | 7가지 핵심 평가 지표(MIA, L2 Distance, Neuron Activation, PPL/LogProb, KL Divergence, MMLU 500샘플, GSM8K 200샘플)를 산출하는 모듈식 라이브러리 함수 모음입니다. |
| **`eval_baselines.py`** | **Baseline Pipeline** | 기준 모델(`1_base`, `2_retain_only`) 전용 평가 스크립트입니다. 상대 지표를 제외한 모델 본연의 절대 지표(Lower/Upper Bound)를 추출하여 `./eval_standard_results/`에 저장합니다. |
| **`run_eval.py`** | **Target Pipeline** | 9개 언러닝 변체 모델 전용 평가 스크립트입니다. Base 및 Retain-only 레퍼런스 모델과의 상대 대조 수치(`_vs_base`, `_vs_retain`)를 포함해 종합 평가 후 `./eval_results/`에 저장합니다. |
| **`download_baselines.py`** | **Utility** | Open Unlearning 리포지토리에서 사전 학습된 Base 및 Retain-only 모델 체크포인트를 로컬 디렉토리(`./checkpoints/`)로 자동 다운로드하는 보조 스크립트입니다. |
| **`summarize_results.py`** | **Analysis Utility** | 평가 완료 후 각 모델별 개별 JSON 파일들(`models/*.json`)과 통합 리포트(`integrated_summary.json`)를 읽어와 Pandas DataFrame 비교표(CSV 및 터미널 출력)로 정형화하는 스크립트입니다. |

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

## 🚀 파이프라인 실행 방법 (Workflow)

### Step 1. 레퍼런스 기준 모델 평가 (Base & Retain-only)
```bash
python eval_standard.py