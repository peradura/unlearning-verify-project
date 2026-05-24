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


if __name__ == "__main__":
    PATH_ORIGINAL = "./model_check/test_orig"
    PATH_UNLEARNED = "./model_check/test_unlearn"

    print("🔄 아키텍처 자동 분석형 뉴런 스캐너 가동...")
    if load_model_cpu(PATH_ORIGINAL, target_type="original") and load_model_cpu(PATH_UNLEARNED,
                                                                                target_type="unlearned"):
        tokenizer = AutoTokenizer.from_pretrained(PATH_ORIGINAL, trust_remote_code=True)
        model_orig = MODEL_REGISTRY["original"]
        model_unlearn = MODEL_REGISTRY["unlearned"]

        # --------------------------------------------------------
        # 🎯 [핵심] 아키텍처 파싱 및 스캔 구간 자동 타겟팅
        # --------------------------------------------------------
        # 모델 오브젝트 내부에서 실제 레이어 모듈 리스트를 자동으로 추출합니다.
        layers_orig = model_orig.model.layers
        layers_unlearn = model_unlearn.model.layers

        total_layers = len(layers_orig)
        print(f"🔍 [아키텍처 확인] 이 모델의 총 레이어 개수: {total_layers}개")

        # 유저님이 원하시는 후반부 영역 비율 설정 (예: 50% 지점부터 90% 지점까지)
        START_PROP = 0.50
        END_PROP = 0.90

        start_idx = int(total_layers * START_PROP)
        end_idx = int(total_layers * END_PROP)
        target_indices = list(range(start_idx, end_idx + 1))

        print(f"🎯 [자동 구간 매핑] 실측 스캔 타겟 레이어 범위: {start_idx}번 ~ {end_idx}번 레이어 (총 {len(target_indices)}개 레이어 집중 포커싱)")

        # 🪝 설정된 구간 내 모든 레이어의 gate_proj에 멀티 훅 걸기
        hooks = []
        for idx in target_indices:
            hooks.append(layers_orig[idx].mlp.gate_proj.register_forward_hook(make_multi_layer_hook("original")))
            hooks.append(layers_unlearn[idx].mlp.gate_proj.register_forward_hook(make_multi_layer_hook("unlearned")))

        # 📂 Forget 데이터 로드
        FORGET_DATA_PATH = "../dataset/forget10_data.json"
        with open(FORGET_DATA_PATH, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        print(f"\n🚀 [화이트박스 뇌지도 스캔] 타겟 구간 데이터 순회 중 (100개 샘플)...")
        for sample in dataset[:100]:
            combined_text = f"{sample['prompt'].strip()} {sample['response'].strip()}"
            inputs = tokenizer(combined_text, return_tensors="pt").to("cpu")
            with torch.no_grad():
                _ = model_orig(**inputs)
                _ = model_unlearn(**inputs)

        # 메모리 보호를 위해 등록한 모든 훅 제거
        for hk in hooks:
            hk.remove()

        # 📊 지표 평균 연산
        orig_mean_intensity = act_store["original"]["total_val"] / act_store["original"]["total_count"]
        unlearn_mean_intensity = act_store["unlearned"]["total_val"] / act_store["unlearned"]["total_count"]

        orig_active_pct = (act_store["original"]["active_count"] / act_store["original"]["total_count"]) * 100
        unlearn_active_pct = (act_store["unlearned"]["active_count"] / act_store["unlearned"]["total_count"]) * 100

        # 구간 전체 벡터의 코사인 유사도 평균 내기
        v_orig = torch.stack(act_store["original"]["vectors"]).mean(dim=0)
        v_unlearn = torch.stack(act_store["unlearned"]["vectors"]).mean(dim=0)
        global_cos_sim = torch.nn.functional.cosine_similarity(v_orig, v_unlearn, dim=0).item()

        # --------------------------------------------------------
        # 🏁 구간 전수 스캔 최종 보고서 출력
        # --------------------------------------------------------
        print("\n" + "=" * 60)
        print(f"🏁 [구간 전수 스캔 뉴런 활성도 최종 분석 리포트]")
        print("=" * 60)
        print(f"📈 분석 영역 비율 : 전체 {total_layers}개 레이어 중 {START_PROP * 100:.0f}% ~ {END_PROP * 100:.0f}% 구간")
        print("-" * 60)
        print(f"📊 오리지널 원본 모델 (Original Model)")
        print(f"  ├ 구간 뉴런 평균 활성 강도 : {orig_mean_intensity:.6f}")
        print(f"  └ 구간 내 핵심 뉴런 활성화 비율 : {orig_active_pct:.2f}%")
        print("-" * 60)
        print(f"📊 언러닝 가중치 모델 (Unlearned Model)")
        print(f"  ├ 구간 뉴런 평균 활성 강도 : {unlearn_mean_intensity:.6f} 🧠")
        print(f"  └ 구간 내 핵심 뉴런 활성화 비율 : {unlearn_active_pct:.2f}%")
        print("-" * 60)
        print(f"🎯 타겟 구간 전체의 뇌 지도 코사인 유사도 : {global_cos_sim:.4f}")
        print(f"  💡 해석: 지식이 올바르게 제거되었다면 이 다중 레이어 유사도가 낮게 꺾여야 합니다.")
        print("=" * 60)