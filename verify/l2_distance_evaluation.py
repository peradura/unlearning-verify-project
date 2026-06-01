import argparse
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_layer_l2_experiment(model_orig, model_unlearn):
    """
    [화이트박스] 레이어별 가중치 수치 실측 (주노드와 동일한 Pure L2 Distance)
    """
    print("\n" + "=" * 65)
    print("🔬 [실험 1] 레이어별 가중치 행렬 Pure L2 Distance 실측 결과")
    print("=========================================================")

    layers_orig = model_orig.model.layers
    layers_unlearn = model_unlearn.model.layers

    layer_l2_results = {}

    for i in range(len(layers_orig)):
        # 🌟 [.float() 제거] 모델이 로드된 원래 데이터 타입(bfloat16 등) 그대로 연산에 활용
        w_orig = layers_orig[i].self_attn.q_proj.weight.data
        w_unlearn = layers_unlearn[i].self_attn.q_proj.weight.data

        # 🌟 [Pure L2 수식 적용] 분모 정규화 과정을 제거하여 0.3~0.4 스케일 복원
        pure_l2_dist = torch.norm(w_orig - w_unlearn, p=2).item()

        layer_l2_results[f"layer_{i:02d}"] = pure_l2_dist
        print(f"   └ [Layer {i:02d}] Attention Q-Projection Pure L2 Distance: {pure_l2_dist:.6f}")

    print("=========================================================")
    return layer_l2_results


# # 터미널에서 단독 실행 시 작동하는 메인 제어 루프
# if __name__ == "__main__":
#     # 1. 파일 경로 설정
#     PATH_ORIGINAL = "../models/before"
#     PATH_UNLEARNED = "../models/after"
#
#     # 가상환경 인프라 세팅 점검 및 출력
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"🖥️ 연산 디바이스: {device.upper()}")
#
#     # 2. 하드디스크 자산 존재 여부 실측 안전장치
#     if not os.path.exists(PATH_ORIGINAL) or not os.path.exists(PATH_UNLEARNED):
#         print(
#             f"❌ [에러] 모델 경로를 찾을 수 없습니다.\n ├ 오리지널 경로: {os.path.abspath(PATH_ORIGINAL)}\n └ 언러닝 경로: {os.path.abspath(PATH_UNLEARNED)}")
#         exit(1)
#
#     # 3. 오리지널(Before) 자산 로드
#     print("\n📦 [1/2] Original 원본 가중치 로딩 가동 중...")
#     model_orig = AutoModelForCausalLM.from_pretrained(
#         PATH_ORIGINAL,
#         # 🌟 주노드에서 쓰시던 정밀도(bfloat16 혹은 원래 저장 규격)로 맞추려면 자동 로드 권장
#         torch_dtype="auto",
#         device_map="auto" if torch.cuda.is_available() else None,
#         low_cpu_mem_usage=True,
#         trust_remote_code=True
#     )
#
#     # 4. 언러닝(After) 자산 로드
#     print("\n📦 [2/2] Unlearned 연구 가중치 로딩 가동 중...")
#     model_unlearn = AutoModelForCausalLM.from_pretrained(
#         PATH_UNLEARNED,
#         torch_dtype="auto",  # 동일하게 auto 설정
#         device_map="auto" if torch.cuda.is_available() else None,
#         low_cpu_mem_usage=True,
#         trust_remote_code=True
#     )
#
#     # 5. 핵심 가중치 행렬 이격 수치 실측 분석 함수 점화
#     results = run_layer_l2_pure_experiment(model_orig, model_unlearn)
#
#     # 6. 상위 변형 변위 레이어(Top 3) 분석 서머리 리포트 빌드
#     top_layers = sorted(results.items(), key=lambda x: x[1], reverse=True)[:3]
#
#     print("\n🎯 [종합 결론] 가중치 변위량 상위 Top 3 레이어")
#     for rank, (layer, val) in enumerate(top_layers, 1):
#         print(f" ├ {rank}등: {layer} (Pure L2 = {val:.6f})")
#     print("=========================================================\n")