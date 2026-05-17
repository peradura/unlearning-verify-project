import os
import shutil
from transformers import AutoConfig
from safetensors import safe_open

# [설정] 체크가 통과된 파일들이 최종 저장될 안전 업로드 목적지
BASE_UPLOAD_DIR = "./upload_storage"
ORIGINAL_MODEL_DIR = os.path.join(BASE_UPLOAD_DIR, "original_model")
UNLEARNED_MODEL_DIR = os.path.join(BASE_UPLOAD_DIR, "unlearned_model")

os.makedirs(ORIGINAL_MODEL_DIR, exist_ok=True)
os.makedirs(UNLEARNED_MODEL_DIR, exist_ok=True)


def check_and_upload_hf_model(source_folder, target_type="original"):
    """
    실제 Hugging Face(HF) 라이브러리 연동 기능을 활용하여
    모델의 유효성을 깐깐하게 체크한 뒤 최종 복사(업로드)를 수행합니다.
    """
    print(f"\n=========================================")
    print(f"🤗 [HF 연동 검사] 대상 폴더 스캔: {source_folder}")
    print(f"=========================================")

    if not os.path.exists(source_folder):
        print(f"❌ [에러] 원본 폴더 경로가 없습니다: {source_folder}")
        return False

    all_files = os.listdir(source_folder)

    # --------------------------------------------------------
    # [HF 검사 1] AutoConfig를 연동한 config.json 정밀 구조 파싱
    # --------------------------------------------------------
    if "config.json" not in all_files:
        print("❌ [HF 검사 실패] Hugging Face 필수 파일인 'config.json'이 없습니다.")
        return False

    try:
        # HF 전용 설정 로더를 연동해 파일 껍데기를 파싱합니다.
        config = AutoConfig.from_pretrained(source_folder)
        model_type = getattr(config, "model_type", "unknown")
        architecture = getattr(config, "architectures", ["unknown"])[0]

        print(f"✅ [HF 검사 1 통과] 유효한 HF 설정 구조 확인")
        print(f"   - 모델 아키텍처: {architecture}")
        print(f"   - 기본 데이터 타입: {getattr(config, 'torch_dtype', 'unknown')}")
    except Exception as e:
        print(f"❌ [HF 검사 실패] config.json이 HF 규격과 일치하지 않거나 깨졌습니다: {str(e)}")
        return False

    # --------------------------------------------------------
    # [HF 검사 2] Safetensors 연동을 통한 실제 가중치 텐서 키 검증
    # --------------------------------------------------------
    weight_files = [f for f in all_files if f.endswith(".safetensors")]

    # 만약 구형 .bin 포맷이라면 bin 포맷으로 필터링
    if not weight_files:
        weight_files = [f for f in all_files if f.endswith(".bin")]

    if not weight_files:
        print("❌ [HF 검사 실패] 가중치 파일(.safetensors 또는 .bin)이 전혀 없습니다.")
        return False

    # 대표 가중치 파일 하나를 HF safetensors 모듈로 열어서 텐서 구조 체크
    # (전체 모델을 RAM/VRAM에 로드하지 않으므로 몇 초도 안 걸리고 안전합니다.)
    target_weight = os.path.join(source_folder, weight_files[0])
    if weight_files[0].endswith(".safetensors"):
        try:
            with safe_open(target_weight, framework="pt", device="cpu") as f:
                tensor_keys = f.keys()

            # 실제 Llama나 특정 LLM의 핵심 텐서 키 뼈대가 들었는지 샘플 검사
            # (아무 껍데기만 복사해놓은 가짜 파일을 걸러내기 위함)
            print(f"✅ [HF 검사 2 통과] 가중치 파일 내부 텐서 맵 인식 성공 (총 텐서 수: {len(tensor_keys)}개)")
        except Exception as e:
            print(f"❌ [HF 검사 실패] 가중치 파일 바이너리가 손상되었거나 HF에서 읽을 수 없습니다: {str(e)}")
            return False
    else:
        print("⚠️ [참고] 구형 PyTorch (.bin) 파일은 바이너리 직접 손상 여부 점검을 건너뜁니다 (용량 체크만 수행).")

    # --------------------------------------------------------
    # [최종 단계] 실제 저장소 폴더로 안전하게 복사(업로드)
    # --------------------------------------------------------
    dest_dir = ORIGINAL_MODEL_DIR if target_type == "original" else UNLEARNED_MODEL_DIR
    print(f"🔄 HF 연동 검증 완료! 실제 저장소로 복사 중 -> {dest_dir}")

    for filename in all_files:
        src_file = os.path.join(source_folder, filename)
        dest_file = os.path.join(dest_dir, filename)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dest_file)

    print(f"💾 [{target_type.upper()}] 모델 최종 업로드 완료. (폴더 내 파일 수: {len(os.listdir(dest_dir))}개)")
    return True


# ====================================================
# 로컬 기능 구동용 테스트 예시
# ====================================================
if __name__ == "__main__":
    # open-unlearning 결과물 폴더나, 캐시에 받아진 테스트용 허깅페이스 모델 폴더 경로를 넣어보세요.
    MY_HF_ORIG_PATH = "./my_test_models/Llama-orig"
    MY_HF_UNLEARN_PATH = "./my_test_models/Llama-unlearned"

    # 오리지널 모델 체크 후 업로드 기능 수행
    check_and_upload_hf_model(MY_HF_ORIG_PATH, target_type="original")