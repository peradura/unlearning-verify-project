import torch
import numpy as np


def run_layer_l2_normalized_experiment(model_orig, model_unlearn):
    """
    [화이트박스] 레이어별 가중치 수치 실측 (정규화된 L2 Distance)
    구조: 유저님 원본 코드의 0번~끝번 레이어 순회 구조 완벽 유지
    수식: 두 번째 코드 메커니즘 반영 -> L2 거리(p=2) / np.sqrt(원소 총 개수)
    """
    print("\n" + "=" * 65)
    print("🔬 [실험 1] 레이어별 가중치 행렬 Normalized L2 Distance 실측 결과")
    print("=========================================================")

    layers_orig = model_orig.model.layers
    layers_unlearn = model_unlearn.model.layers

    for i in range(len(layers_orig)):
        # 1. 각 레이어의 Q-Projection 가중치 데이터 추출 (연산 정밀도를 위해 float32로 변환)
        w_orig = layers_orig[i].self_attn.q_proj.weight.data.float()
        w_unlearn = layers_unlearn[i].self_attn.q_proj.weight.data.float()

        # 2. 순수 L2 Distance 계산 (p=2: 오리지널 L2 절대 거리)
        l2_dist = torch.norm(w_orig - w_unlearn, p=2).item()

        # 3. 🎯 L2 정규화 매커니즘 이식 (총 가중치 원소 수의 제곱근으로 나눔)
        # 💡 의미: 가중치 행렬의 크기에 왜곡되지 않는 "원소당 평균 RMS 변동 오차"
        num_elements = w_orig.numel()
        norm_l2 = l2_dist / np.sqrt(num_elements) if num_elements > 0 else 0

        print(f"   └ [Layer {i:02d}] Attention Q-Projection Normalized L2 Distance: {norm_l2:.6f}")

    print("=========================================================")