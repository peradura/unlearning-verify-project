# 🤖 TOFU Unlearning Evaluation Pipeline for Llama-3.2-1B-Instruct

이 프로젝트는 **Llama-3.2-1B-Instruct** 모델을 대상으로 **TOFU 데이터셋(`forget10`)** 기반의 3가지 머신 언러닝(Machine Unlearning) 기법(**GradDiff, AltPO, NPO**)을 수행하고 평가하기 위한 모듈식 파이프라인입니다.

언러닝 학습, 재학습 및 4-bit 양자화를 통해 총 11개 모델 변체(Base, Retain-only, FP16 언러닝 3종, 50% Forget 재학습 3종, 4-bit 양자화 3종)를 생성하고 7가지 표준 지표를 통해 대조·분석합니다.

---

## 📜 파일 구조 및 스크립트 설명

| 파일명 | 구분 | 설명 및 역할 |
| :--- | :--- | :--- |
| **`evaluator.py`** | **Core Library** | 7가지 핵심 평가 지표(MIA, $L_2$ Distance, Neuron Activation, PPL/LogProb, KL Divergence, MMLU 500샘플, GSM8K 200샘플)를 산출하는 모듈식 라이브러리 함수 모음입니다. |
| **`relearning.py`** | **Train Pipeline** | HF 허브(open-unlearning)의 FP16 언러닝 모델(GradDiff, AltPO, NPO)에 Forget 데이터 전반부 50%를 재학습시켜 재학습 모델 3종을 생성합니다. 후반부 50%는 평가 전용으로 남깁니다. |
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

모든 스크립트는 `analysis/` 디렉토리 안에서 실행하며, `--help`로 인자를 확인할 수 있습니다. 인자를 생략하면 GradDiff 기준 기본값으로 동작합니다.

```bash
# 1. 기준 모델 2종(Base, Retain-only) 절대 지표 추출 → ./eval_standard_results/
python eval_standard.py --model_name 1_base --model_path open-unlearning/pos_tofu_Llama-3.2-1B-Instruct_full_lr2e-05_wd0.01_epoch10
python eval_standard.py --model_name 2_retain_only --model_path open-unlearning/neg_tofu_Llama-3.2-1B-Instruct_retain90_forget10_pert_lr1e-05_wd0.01_epoch10

# 2. FP16 언러닝 모델 3종 평가 (HF 허브 체크포인트 그대로 사용) → ./eval_unlearned_results/
python eval_unlearned.py --model_name 3_graddiff_fp16 --model_path open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch5

# 3. 50% Forget 재학습 모델 생성 후 평가
python relearning.py --model_path <FP16 언러닝 모델> --save_dir ./checkpoints/4_graddiff_relearned_50
python eval_unlearned.py --model_name 4_graddiff_relearned_50 --model_path ./checkpoints/4_graddiff_relearned_50

# 4. 4-bit 양자화 모델 생성 후 평가 (--quantized 필수)
python quantization.py --model_path <FP16 언러닝 모델> --save_dir ./checkpoints/5_graddiff_quant_4bit
python eval_unlearned.py --quantized --model_name 5_graddiff_quant_4bit --model_path ./checkpoints/5_graddiff_quant_4bit
```

모델 번호 규칙: 1 base, 2 retain_only, 3/4/5 GradDiff(fp16/재학습50%/양자화), 6/7/8 AltPO, 9/10/11 NPO, 12/13/14 GradDiff/AltPO/NPO 재학습 20% (`relearning.py --relearn_frac 0.2`).
