import json
import os
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# 🧠 모델 레지스트리 (CPU 상주용)
MODEL_REGISTRY = {"original": None, "unlearned": None}

# 🪝 전 구간 뉴런 활성 강도를 누적할 딕셔너리
act_store = {
    "original": {"total_val": 0.0, "active_count": 0, "total_count": 0, "vectors": []},
    "unlearned": {"total_val": 0.0, "active_count": 0, "total_count": 0, "vectors": []}
}


def make_multi_layer_hook(model_type, thresh=0.01):
    """지정된 구간 내 모든 레이어에서 공통으로 호출되어 누적 연산하는 훅"""

    def hook_fn(module, input, output):
        # output: [batch, seq_len, hidden_dim]
        act_val = output.detach().float().mean(dim=(0, 1)).abs()

        # 1. 활성 세기 합산 및 뉴런 개수 카운트
        act_store[model_type]["total_val"] += act_val.sum().item()
        act_store[model_type]["active_count"] += (act_val > thresh).sum().item()
        act_store[model_type]["total_count"] += act_val.numel()

        # 2. 코사인 유사도 측정을 위해 레이어별 평균 벡터 보관
        act_store[model_type]["vectors"].append(act_val)

    return hook_fn


def load_model_cpu(source_folder, target_type="original"):
    if not os.path.exists(source_folder):
        print(f"❌ 경로 부재: {source_folder}")
        return False
    try:
        model = AutoModelForCausalLM.from_pretrained(
            source_folder, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True
        )
        model.eval()
        MODEL_REGISTRY[target_type] = model
        return True
    except Exception as e:
        print(f"❌ 로드 실패: {e}")
        return False

