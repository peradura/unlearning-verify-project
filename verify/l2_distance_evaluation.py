import torch
import numpy as np


def run_layer_l1_normalized_experiment(model_orig, model_unlearn):
    """
    [화이트박스] 레이어별 가중치 수치 실측 (정규화된 L1 Distance)
    구조: 첫 번째 코드의 레이어 루프 구조 체택
    수식: 두 번째 코드의 원소 수(numel) 기반 정규화 + L1(절대값 편차) 연산
    """
    print("\n" + "=" * 65)
    print("🔬 [실험] 레이어별 가중치 행렬 Normalized L1 Distance 실측 결과")
    print("=========================================================")

    layers_orig = model_orig.model.layers
    layers_unlearn = model_unlearn.model.layers

    # 결과를 상위 파이프라인으로 넘겨줄 수 있도록 변수 수납고 세팅
    layer_l1_results = {}

    for i in range(len(layers_orig)):
        # 1. 10번을 포함한 각 레이어의 Q-Projection 가중치 데이터 로드 (.float()로 정밀도 고정)
        w_orig = layers_orig[i].self_attn.q_proj.weight.data.float()
        w_unlearn = layers_unlearn[i].self_attn.q_proj.weight.data.float()

        # 2. L1 Distance 계산 (p=1: 원소별 차이의 절대값 총합)
        l1_dist = torch.norm(w_orig - w_unlearn, p=1).item()

        # 3. L1 정규화 매커니즘 이식 (총 L1 거리를 가중치 원소 총 개수로 나눔)
        # 💡 의미: "해당 레이어 가중치 소수점 자리 하나당 평균적으로 발생한 수치적 변동량"
        num_elements = w_orig.numel()
        norm_l1 = l1_dist / num_elements if num_elements > 0 else 0

        # 변수 저장 및 터미널 모니터링 출력
        layer_l1_results[f"layer_{i:02d}"] = norm_l1
        print(f"   └ [Layer {i:02d}] Attention Q-Proj Normalized L1: {norm_l1:.6f}")

    print("=========================================================")

    # 다른 변수에 바로 바인딩할 수 있도록 딕셔너리 리턴
    return layer_l1_results