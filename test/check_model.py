import os
import re
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# 🧠 모델 레지스트리 (CPU 상주용)
MODEL_REGISTRY = {
    "original": None,
    "unlearned": None
}


def load_and_verify_model_pipeline_cpu(source_folder, target_type="original"):
    print("\n" + "=" * 50)
    print(f"🖥️ [CPU 모드] 모델 검증 및 RAM 상주 시작 [{target_type.upper()}]")
    print(f"📂 경로: {source_folder}")
    print("=" * 50)

    # [0단계] 폴더 체크
    if not os.path.exists(source_folder):
        print(f"❌ [오류] 지정된 모델 폴더 경로가 존재하지 않습니다: {source_folder}")
        return False

    # -----------------------------------------------------
    # 1️⃣ 1차 검증 (Config)
    # -----------------------------------------------------
    print("\n🧪 [1단계] config.json 구조 파싱 및 아키텍처 검증")
    try:
        config = AutoConfig.from_pretrained(source_folder)
        model_type = getattr(config, "model_type", "model")
        model_type = re.sub(r"[^a-zA-Z0-9_-]", "_", model_type)
        torch_dtype = torch.float16  # CPU 메모리 절약을 위한 반정밀도 세팅
        print(f"   ├ 아키텍처 확인: {model_type}")
        print(f"   └ 연산 포맷 매핑: {torch_dtype} (CPU)")
    except Exception as e:
        print(f"❌ [1단계 실패] config 파싱 에러: {e}")
        return False

    # -----------------------------------------------------
    # 2️⃣ 2차 검증 (Tokenizer)
    # -----------------------------------------------------
    print("\n🧪 [2단계] 아키텍처 기반 토크나이저(Tokenizer) 자동 로드")
    try:
        tokenizer = AutoTokenizer.from_pretrained(source_folder, trust_remote_code=True)
        print(f"   ├ 어휘 사전 크기(Vocab Size): {len(tokenizer)}개")
        print(f"   └ 특수 토큰(BOS ID) 검증: {tokenizer.bos_token_id}")
    except Exception as e:
        print(f"❌ [2단계 실패] 토크나이저 로드 에러: {e}")
        return False

    # -----------------------------------------------------
    # 3️⃣ 3차 검증 (Model Load)
    # -----------------------------------------------------
    print("\n🧪 [3단계] 진짜 가중치(.safetensors) 시스템 RAM 로딩")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            source_folder,
            torch_dtype=torch_dtype,
            device_map="cpu",  # 핵심: CPU 메인 RAM으로 강제 타겟팅
            low_cpu_mem_usage=True
        )
        model.eval()

        MODEL_REGISTRY[target_type] = model
        assigned_device = next(model.parameters()).device

        print(f"✅ [3단계 통과] {target_type} 모델 CPU 준비 완수!")
        print(f"   └ 상주 디바이스 레지스트리 포트: {assigned_device}")
        return True
    except Exception as e:
        print(f"❌ [3단계 실패] 가중치 행렬 로딩 중 에러 발생: {e}")
        return False


def run_layer_l2_experiment():
    """진짜 CPU 메모리에 올라간 가중치 행렬 간의 차이 실측 계산"""
    model_orig = MODEL_REGISTRY["original"]
    model_unlearn = MODEL_REGISTRY["unlearned"]

    if model_orig is None or model_unlearn is None:
        print("\n❌ [실험 실패] 두 모델이 모두 레지스트리에 상주해 있어야 합니다.")
        return

    print("\n" + "=" * 50)
    print("🔬 [CPU 실험 가동] 레이어별 가중치 행렬 L2 Distance 실측 결과")
    print("=" * 50)

    layers_orig = model_orig.model.layers
    layers_unlearn = model_unlearn.model.layers

    # Llama 3.2 1B 모델의 16개 레이어를 순회하며 실측 연산 수행
    for i in range(len(layers_orig)):
        w_orig = layers_orig[i].self_attn.q_proj.weight.data
        w_unlearn = layers_unlearn[i].self_attn.q_proj.weight.data

        # PyTorch CPU 스레드를 이용한 실제 수치 행렬 연산
        l2_dist = torch.norm(w_orig - w_unlearn, p=2).item()
        print(f"   └ [Layer {i:02d}] Attention Q-Projection L2 Distance: {l2_dist:.6f}")


# =========================================================
# 🚨 멈춤 방지용 메인 실행부 (여기가 꼭 있어야 터미널에 찍힙니다)
# =========================================================
if __name__ == "__main__":
    # 다운로드 받은 실제 모델들의 상대 경로 설정
    PATH_ORIGINAL = "./model_check/test_orig"
    PATH_UNLEARNED = "./model_check/test_unlearn"

    # 1. 오리지널 원본 모델 검증 및 로드
    orig_ok = load_and_verify_model_pipeline_cpu(PATH_ORIGINAL, target_type="original")

    # 2. 언러닝 기법 적용 모델 검증 및 로드
    unlearn_ok = load_and_verify_model_pipeline_cpu(PATH_UNLEARNED, target_type="unlearned")

    # 3. 둘 다 로드 성공 시 진짜 가중치 행렬 뺄셈 연산 수행
    if orig_ok and unlearn_ok:
        # 기존 L2 실험 실행
        run_layer_l2_experiment()
        print("\n🎉 모든 CPU 기반 실험 프로세스가 성공적으로 마무리되었습니다!")

        # --------------------------------------------------------
        # 🎯 [추가] 말을 제대로 하는지 단일 프롬프트로 최종 검증
        # --------------------------------------------------------
        print("\n" + "=" * 60)
        print("🗣️  [텍스트 생성 능력 검증] 모델이 정상적인 문장을 구사하는지 체크")
        print("=" * 60)

        # 모델의 일반적인 언어 유틸리티가 무너지지 않았는지 상식 질문 주입
        test_prompt = "What are the core benefits of regular cardiovascular exercise?"

        # 레지스트리에서 모델 꺼내오기 및 토크나이저 재로드
        model_orig = MODEL_REGISTRY["original"]
        model_unlearn = MODEL_REGISTRY["unlearned"]

        # 가벼운 토크나이저 로드 (기본 설정 유지)
        tokenizer = AutoTokenizer.from_pretrained(PATH_ORIGINAL, trust_remote_code=True)
        inputs = tokenizer(test_prompt, return_tensors="pt").to("cpu")

        # 1. 원본 모델 테스트
        print("\n🔍 [1] 오리지널 원본 모델 답변:")
        with torch.no_grad():
            gen_orig = model_orig.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7)
        print(f"👉 {tokenizer.decode(gen_orig[0], skip_special_tokens=True)}")

        print("-" * 60)

        # 2. 언러닝 모델 테스트 (말이 깨지는지 집중 확인)
        print("🔍 [2] 언러닝 완료 모델 답변:")
        with torch.no_grad():
            gen_unlearn = model_unlearn.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7)
        print(f"👉 {tokenizer.decode(gen_unlearn[0], skip_special_tokens=True)}")
        print("=" * 60)

    else:
        print("\n❌ 모델 로딩 단계를 완수하지 못해 지표 추출 실험을 건너뜁니다.")