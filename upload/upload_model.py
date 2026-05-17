import os
import re
import json
import shutil
import tempfile
import torch

from transformers import (
    AutoConfig,
    AutoModelForCausalLM
)

from safetensors import safe_open


# =========================================================
# 📂 최상위 저장소
# =========================================================
BASE_MODEL_DIR = "./model"


# =========================================================
# 🧠 GPU 상주 모델 레지스트리
# =========================================================
MODEL_REGISTRY = {
    "original": None,
    "unlearned": None
}


# =========================================================
# 📦 tokenizer whitelist
# =========================================================
TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "added_tokens.json"
}


# =========================================================
# 🔒 safetensors 무결성 검사
# =========================================================
def validate_safetensor_file(path):

    try:

        with safe_open(
            path,
            framework="pt",
            device="cpu"
        ) as f:

            keys = list(f.keys())

        if len(keys) == 0:
            return False

        return True

    except Exception as e:

        print(f"❌ safetensors 검증 실패: {e}")

        return False


# =========================================================
# 📏 모델 크기 추론 (b 단위)
# =========================================================
def get_model_size_string(config):

    num_params = getattr(
        config,
        "num_parameters",
        None
    )

    if num_params is not None:

        billions = round(num_params / 1e9)

        return f"{billions}b"

    hidden_size = getattr(
        config,
        "hidden_size",
        0
    )

    if hidden_size == 2048:
        return "1b"

    elif hidden_size == 3072:
        return "3b"

    elif hidden_size == 4096:
        return "7b"

    elif hidden_size == 5120:
        return "13b"

    return "unknown"


# =========================================================
# 📚 shard completeness 검사
# =========================================================
def validate_shard_completeness(
    source_folder,
    all_files
):

    index_file = None

    for f in all_files:

        if f.endswith(".index.json"):

            index_file = f
            break

    if index_file is None:
        return True

    try:

        index_path = os.path.join(
            source_folder,
            index_file
        )

        with open(
            index_path,
            "r",
            encoding="utf-8"
        ) as f:

            index_data = json.load(f)

        required_shards = set(
            index_data["weight_map"].values()
        )

        missing = []

        for shard in required_shards:

            if shard not in all_files:

                missing.append(shard)

        if len(missing) > 0:

            print("❌ shard 누락 발견")

            for m in missing:
                print(f"   └ {m}")

            return False

        print("✅ shard completeness 검사 통과")

        return True

    except Exception as e:

        print(f"❌ shard 검사 실패: {e}")

        return False


# =========================================================
# 🔍 파일 사전 검증
# =========================================================
def validate_model_before_upload(
    source_folder,
    all_files
):

    # -----------------------------------------------------
    # config.json 필수
    # -----------------------------------------------------
    if "config.json" not in all_files:

        print("❌ config.json 없음")

        return False

    # -----------------------------------------------------
    # safetensors 검사
    # -----------------------------------------------------
    safetensor_found = False

    for filename in all_files:

        src_file = os.path.join(
            source_folder,
            filename
        )

        if filename.endswith(".safetensors"):

            safetensor_found = True

            print(
                f"🔍 safetensors 검사 중: "
                f"{filename}"
            )

            if not validate_safetensor_file(
                src_file
            ):

                print(
                    "❌ 손상된 safetensors 발견"
                )

                return False

    if not safetensor_found:

        print("❌ safetensors 없음")

        return False

    # -----------------------------------------------------
    # shard 검사
    # -----------------------------------------------------
    if not validate_shard_completeness(
        source_folder,
        all_files
    ):
        return False

    return True


# =========================================================
# 🚀 GPU 상주 로딩
# =========================================================
def upload_and_keep_on_gpu(
    source_folder,
    model_key,
    model_type,
    model_size
):

    print(
        f"\n🚀 GPU 상주 로딩 시작 "
        f"({model_type}_{model_size})"
    )

    if not torch.cuda.is_available():

        print("❌ GPU 없음")

        return False

    try:

        # -------------------------------------------------
        # dtype 자동 선택
        # -------------------------------------------------
        torch_dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

        print(
            f"   └ dtype: {torch_dtype}"
        )

        print(
            "   └ VRAM 업로드 중..."
        )

        # -------------------------------------------------
        # 실제 GPU 업로드
        # -------------------------------------------------
        model = AutoModelForCausalLM.from_pretrained(
            source_folder,
            torch_dtype=torch_dtype,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        model.eval()

        # -------------------------------------------------
        # registry 등록
        # -------------------------------------------------
        MODEL_REGISTRY[model_key] = model

        print("✅ GPU 상주 완료")

        print(
            f"   └ registry key: {model_key}"
        )

        print(
            f"   └ device: {model.device}"
        )

        return True

    except torch.cuda.OutOfMemoryError:

        print("❌ GPU OOM")

        return False

    except Exception as e:

        print(
            f"❌ GPU 로딩 실패: {e}"
        )

        return False


# =========================================================
# 🗂️ 메인 함수
# =========================================================
def check_and_upload_hf_model(
    source_folder,
    target_type="original"
):

    print("\n=========================================")
    print("🤗 HF 모델 검증 및 업로드 시작")
    print(f"📂 source: {source_folder}")
    print("=========================================")

    # -----------------------------------------------------
    # source 검사
    # -----------------------------------------------------
    if not os.path.exists(source_folder):

        print("❌ source folder 없음")

        return False

    all_files = os.listdir(source_folder)

    # =====================================================
    # 1️⃣ 파일 검증
    # =====================================================
    print("\n🧪 [1단계] 파일 검증 시작")

    if not validate_model_before_upload(
        source_folder,
        all_files
    ):

        print("\n❌ 검증 실패")
        print("❌ 폴더 생성 안됨")
        print("❌ GPU 업로드 안됨")

        return False

    print("✅ 파일 검증 완료")

    # =====================================================
    # 2️⃣ config 파싱
    # =====================================================
    try:

        config = AutoConfig.from_pretrained(
            source_folder
        )

        model_type = getattr(
            config,
            "model_type",
            "model"
        )

        model_type = re.sub(
            r"[^a-zA-Z0-9_-]",
            "_",
            model_type
        )

        model_size = get_model_size_string(
            config
        )

        base_identity_name = (
            f"{model_type}_{model_size}"
        )

    except Exception as e:

        print(f"❌ config 파싱 실패: {e}")

        return False

    # =====================================================
    # 3️⃣ GPU 상주 로딩
    # =====================================================
    print("\n🧪 [2단계] GPU 상주 로딩")

    if not upload_and_keep_on_gpu(
        source_folder=source_folder,
        model_key=target_type,
        model_type=model_type,
        model_size=model_size
    ):

        print("\n❌ GPU 상주 실패")

        return False

    print("✅ GPU 상주 완료")

    # =====================================================
    # 4️⃣ 저장 경로
    # =====================================================
    if target_type == "original":

        state_folder = "original"
        suffix = "org"

    else:

        state_folder = "unlearned"
        suffix = "unlearned"

    pth_folder_name = (
        f"{base_identity_name}_{suffix}"
    )

    final_pth_dir = os.path.join(
        BASE_MODEL_DIR,
        "pth",
        state_folder,
        pth_folder_name
    )

    config_dir = os.path.join(
        BASE_MODEL_DIR,
        "config",
        base_identity_name
    )

    tokenizer_dir = os.path.join(
        BASE_MODEL_DIR,
        "tokenizer",
        base_identity_name
    )

    # =====================================================
    # overwrite 방지
    # =====================================================
    original_final_dir = final_pth_dir

    version = 1

    while os.path.exists(final_pth_dir):

        version += 1

        final_pth_dir = (
            f"{original_final_dir}_v{version}"
        )

    # =====================================================
    # 5️⃣ temp upload
    # =====================================================
    temp_root = tempfile.mkdtemp(
        prefix="upload_tmp_"
    )

    temp_pth_dir = os.path.join(
        temp_root,
        "pth"
    )

    temp_config_dir = os.path.join(
        temp_root,
        "config"
    )

    temp_tokenizer_dir = os.path.join(
        temp_root,
        "tokenizer"
    )

    os.makedirs(temp_pth_dir)
    os.makedirs(temp_config_dir)
    os.makedirs(temp_tokenizer_dir)

    try:

        # -------------------------------------------------
        # 파일 분류 복사
        # -------------------------------------------------
        for filename in all_files:

            src_file = os.path.join(
                source_folder,
                filename
            )

            if (
                os.path.islink(src_file)
                or not os.path.isfile(src_file)
            ):
                continue

            # weight
            if (
                filename.endswith(".safetensors")
                or filename.endswith(".bin")
                or filename.endswith(".index.json")
            ):

                dest_file = os.path.join(
                    temp_pth_dir,
                    filename
                )

            # tokenizer
            elif filename in TOKENIZER_FILES:

                dest_file = os.path.join(
                    temp_tokenizer_dir,
                    filename
                )

            # config
            else:

                dest_file = os.path.join(
                    temp_config_dir,
                    filename
                )

            shutil.copy2(
                src_file,
                dest_file
            )

        # -------------------------------------------------
        # 최종 디렉토리 생성
        # -------------------------------------------------
        os.makedirs(
            os.path.dirname(final_pth_dir),
            exist_ok=True
        )

        os.makedirs(
            config_dir,
            exist_ok=True
        )

        os.makedirs(
            tokenizer_dir,
            exist_ok=True
        )

        # -------------------------------------------------
        # weight 이동
        # -------------------------------------------------
        shutil.move(
            temp_pth_dir,
            final_pth_dir
        )

        # -------------------------------------------------
        # config 이동
        # -------------------------------------------------
        for f in os.listdir(temp_config_dir):

            src = os.path.join(
                temp_config_dir,
                f
            )

            dst = os.path.join(
                config_dir,
                f
            )

            if not os.path.exists(dst):

                shutil.move(src, dst)

        # -------------------------------------------------
        # tokenizer 이동
        # -------------------------------------------------
        for f in os.listdir(temp_tokenizer_dir):

            src = os.path.join(
                temp_tokenizer_dir,
                f
            )

            dst = os.path.join(
                tokenizer_dir,
                f
            )

            if not os.path.exists(dst):

                shutil.move(src, dst)

        print("\n=========================================")
        print("✅ 검증 + GPU 상주 + 저장 완료")
        print("=========================================")

        print(f"\n📂 weight 저장:")
        print(f"   └ {final_pth_dir}")

        print(f"\n📂 config 저장:")
        print(f"   └ {config_dir}")

        print(f"\n📂 tokenizer 저장:")
        print(f"   └ {tokenizer_dir}")

        return True

    except Exception as e:

        print(f"\n❌ 저장 실패: {e}")

        if os.path.exists(final_pth_dir):

            shutil.rmtree(
                final_pth_dir,
                ignore_errors=True
            )

        return False

    finally:

        if os.path.exists(temp_root):

            shutil.rmtree(
                temp_root,
                ignore_errors=True
            )


# =========================================================
# 🧪 테스트
# =========================================================
if __name__ == "__main__":

    # -----------------------------------------------------
    # ORIGINAL 모델 업로드 + GPU 상주
    # -----------------------------------------------------
    TEMP_SOURCE_ORIG = (
        "./model_check/test_orig"
    )

    check_and_upload_hf_model(
        TEMP_SOURCE_ORIG,
        target_type="original"
    )

    # -----------------------------------------------------
    # UNLEARNED 모델 업로드 + GPU 상주
    # -----------------------------------------------------
    TEMP_SOURCE_UNLEARN = (
        "./model_check/test_unlearn"
    )

    check_and_upload_hf_model(
        TEMP_SOURCE_UNLEARN,
        target_type="unlearned"
    )

    # -----------------------------------------------------
    # REGISTRY 상태 출력
    # -----------------------------------------------------
    print("\n=========================================")
    print("🧠 현재 GPU 상주 Registry")
    print("=========================================")

    print("\n[ORIGINAL]")
    print(MODEL_REGISTRY["original"])

    print("\n[UNLEARNED]")
    print(MODEL_REGISTRY["unlearned"])