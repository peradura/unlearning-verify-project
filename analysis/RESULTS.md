# TOFU forget10 언러닝 11개 모델 관측 결과 (2026-11-11)

Llama-3.2-1B-Instruct 기준 모델 2종과 언러닝 변체 9종(3 기법 × FP16 / 50% 재학습 / 4-bit 양자화)을
`analysis/` 파이프라인으로 평가한 결과와 해석. 원본 수치는 `eval_standard_results/`, `eval_unlearned_results/`의 JSON.

## 실행 조건

- 코드: 커밋 `fdebf3d` (버그 수정 `9c418c1` 포함). 실행 명령은 `readme.md`의 Workflow와 동일.
- 환경: H200 NVL 2장, Python 3.12, torch 2.15, transformers 5.22, datasets 5.2, bitsandbytes 0.50.
- 평가 데이터: TOFU forget10 후반 200개(forget), retain90 앞 200개(retain), MMLU test 앞 500개, GSM8K test 앞 200개. 모두 고정 슬라이스라 재현 가능.
- 재학습: forget10 전반 200개, 3 epoch, lr 1e-5, batch 4, bf16. 양자화: bnb NF4 4-bit, double quant.
- 총 소요 약 38분(GPU 2장 병렬).

## 결과표

| model | MIA | F-PPL | F-logP | MMLU | GSM8K | L2/base | KL/base | KL/retain | MIAgap |
|---|---|---|---|---|---|---|---|---|---|
| 1_base | 0.575 | 7.008 | -1.947 | 0.412 | 0.210 | - | - | - | - |
| 2_retain_only | 0.410 | 14.855 | -2.698 | 0.410 | 0.210 | - | - | - | - |
| 3_graddiff_fp16 | 0.545 | 5.062 | -1.622 | 0.404 | 0.240 | 11.115 | 11.989 | 42.938 | 0.135 |
| 4_graddiff_relearned_50 | 0.505 | 3.088 | -1.128 | 0.412 | 0.265 | 11.722 | 61.909 | 100.4 | 0.095 |
| 5_graddiff_quant_4bit | 0.560 | 5.690 | -1.739 | 0.372 | 0.175 | - | 16.851 | 37.570 | 0.150 |
| 6_altpo_fp16 | 0.242 | 44.498 | -3.795 | 0.408 | 0.200 | 12.281 | 26.902 | 54.631 | 0.167 |
| 7_altpo_relearned_50 | 0.260 | 11.643 | -2.455 | 0.436 | 0.220 | 12.993 | 50.356 | 65.473 | 0.150 |
| 8_altpo_quant_4bit | 0.240 | 38.603 | -3.653 | 0.366 | 0.180 | - | 31.267 | 51.254 | 0.170 |
| 9_npo_fp16 | 0.330 | 8.493 | -2.139 | 0.410 | 0.210 | 11.177 | 18.712 | 55.132 | 0.080 |
| 10_npo_relearned_50 | 0.297 | 4.706 | -1.549 | 0.414 | 0.210 | 11.792 | 56.252 | 88.312 | 0.112 |
| 11_npo_quant_4bit | 0.323 | 9.699 | -2.272 | 0.366 | 0.170 | - | 22.773 | 47.653 | 0.087 |

열 설명: MIA = forget/retain loss 임계값 공격 정확도(0.5에 가까울수록 구분 불가). F-PPL / F-logP = forget 테스트셋 perplexity / 평균 log-prob(높을수록·낮을수록 더 잊음). L2/base = base 대비 전체 파라미터 L2 거리(양자화 모델은 미계산). KL/base, KL/retain = retain 텍스트에서 base 또는 retain-only 모델과의 출력 분포 KL. MIAgap = retain-only 모델 MIA와의 절대 차.

## 해석

**기법 간 비교 (FP16)**

- AltPO가 forget을 가장 강하게 지움(F-PPL 44.5, MIA 0.24). retain-only 모델(F-PPL 14.9)보다도 forget PPL이 3배 높아 "모르는 상태"를 넘어 "적극적으로 틀리는 상태"에 가깝다. 대신 KL/base 26.9로 base와 가장 멀어졌다.
- NPO는 중간(F-PPL 8.5, MIA 0.33). retain-only와 MIA 격차가 0.08로 가장 작아 "처음부터 안 배운 모델"에 가장 가까운 분포.
- GradDiff는 forget PPL이 base보다 **낮다**(5.06 < 7.01). 이 체크포인트는 forget 억제가 되지 않은 상태로 보이며, MIA도 base(0.575)와 거의 같다. open-unlearning 허브의 GradDiff 변형 중 다른 하이퍼파라미터 체크포인트를 써야 할 가능성이 있다.
- 일반 성능(MMLU·GSM8K)은 세 기법 모두 base와 1%p 안쪽으로 유지.

**50% 재학습 (forget 전반 200개로 3 epoch)**

- 세 기법 모두 forget 지식이 상당 부분 복구됨: AltPO 44.5 → 11.6, NPO 8.5 → 4.7, GradDiff 5.1 → 3.1. 평가는 재학습에 쓰지 않은 후반 200개로 했으므로, 같은 저자·주제의 일부만 다시 보여줘도 나머지가 되살아나는 일반화 복구다.
- AltPO는 재학습 후에도 retain-only 수준(F-PPL 11.6 vs 14.9)에 머물러 복구 저항이 가장 크다. MIA도 0.26으로 유지.
- 재학습 모델의 KL/base가 50~62로 크게 뛴다. forget 200개로만 학습해 분포가 쏠린 결과이며, retain 성능(MMLU)은 오히려 소폭 올랐다.

**4-bit 양자화**

- forget 지표는 FP16과 거의 같다(AltPO 44.5 → 38.6, NPO 8.5 → 9.7, GradDiff 5.1 → 5.7). 양자화로 언러닝이 풀리는 현상은 이 설정에서는 관찰되지 않았다.
- 대신 일반 성능이 일관되게 떨어진다: MMLU 약 -4%p(0.41 → 0.37), GSM8K 약 -3~6%p. 세 기법에 공통이므로 양자화 자체의 손실.

## 해석 시 주의

- KL 값은 토큰별 KL의 **시퀀스 합**이다(`batchmean`이 배치 크기 1로만 나눔). 실측상 토큰당 평균은 약 0.26 수준. 절대값 비교보다 모델 간 상대 비교로만 쓸 것.
- MIA의 non-member는 retain90(학습에 쓰인 데이터)이다. 따라서 이 MIA는 "forget과 retain의 loss 분리도"를 재는 것이며, 0.5라고 해서 프라이버시가 보장된다는 뜻은 아니다. 엄밀한 MIA에는 학습에 쓰지 않은 holdout이 필요.
- L2는 fp16으로 계산해 원소의 약 41%에서 제곱 차이가 언더플로된다(fp32 재계산 시 11.11 → 11.26). 순위는 유지되나 절대값은 과소.
- MMLU는 선택지 토큰을 `"A"`(공백 없음)로 비교한다. 모델이 실제로 내는 `" A"`로 바꾸면 base 기준 0.412 → 0.432. 모델 간 비교에는 영향이 작다.
- GSM8K는 `max_new_tokens=256`이라 긴 풀이가 잘리는 샘플이 있다. 정답 파서는 `9c418c1`에서 꼬리 마침표 문제를 고친 버전(base 0.12 → 0.21).

## 재현

```bash
cd analysis
python eval_standard.py --model_name 1_base --model_path open-unlearning/pos_tofu_Llama-3.2-1B-Instruct_full_lr2e-05_wd0.01_epoch10
python eval_unlearned.py --model_name 6_altpo_fp16 --model_path open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.1_alpha1_epoch10
# 나머지 명령은 readme.md Workflow 참고
```
