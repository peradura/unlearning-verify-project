import torch
import numpy as np

def run_layer_l2_normalized_experiment(model_orig, model_unlearn):
    """
    [화이트박스] 레이어별 가중치 수치 실측 (정규화된 L2 Distance)
    """
    print("\n" + "=" * 65)
    print("🔬 [실험 1] 레이어별 가중치 행렬 Normalized L2 Distance 실측 결과")
    print("=========================================================")

    layers_orig = model_orig.model.layers
    layers_unlearn = model_unlearn.model.layers

    layer_l2_results = {}

    for i in range(len(layers_orig)):
        w_orig = layers_orig[i].self_attn.q_proj.weight.data.float()
        w_unlearn = layers_unlearn[i].self_attn.q_proj.weight.data.float()

        l2_dist = torch.norm(w_orig - w_unlearn, p=2).item()
        num_elements = w_orig.numel()
        norm_l2 = l2_dist / torch.norm(w_orig, p=2).item() if torch.norm(w_orig, p=2).item() > 0 else 0

        layer_l2_results[f"layer_{i:02d}"] = norm_l2
        print(f"   └ [Layer {i:02d}] Attention Q-Projection Normalized L2 Distance: {norm_l2:.6f}")

    print("=========================================================")
    return layer_l2_results