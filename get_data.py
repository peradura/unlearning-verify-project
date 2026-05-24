import json
import os
from datasets import load_dataset


def load_and_save_tofu_datasets():
    print("========================================================")
    print("🚀 'locuslab/TOFU' 데이터셋 로딩 및 로컬 저장 시작")
    print("========================================================")

    # 📁 데이터를 저장할 로컬 폴더 경로 설정 (없으면 자동으로 만듭니다)
    output_dir = "./tofu_local_data"
    os.makedirs(output_dir, exist_ok=True)

    # 1. ❌ 지워야 할 10% 타겟 데이터셋 로드
    print("\n📦 [1/2] Forget 10% 데이터셋 처리 중...")
    try:
        raw_forget10 = load_dataset(path="locuslab/TOFU", name="forget10", split="train")
        forget10_dataset = [
            {
                "prompt": sample["question"].strip(),
                "response": sample["answer"].strip()
            }
            for sample in raw_forget10
        ]

        # 💾 로컬 폴더에 JSON 파일로 수납 저장
        forget_file_path = os.path.join(output_dir, "forget10_data.json")
        with open(forget_file_path, "w", encoding="utf-8") as f:
            json.dump(forget10_dataset, f, ensure_ascii=False, indent=2)

        print(f"   ├ ✅ 메모리 상주 완료! 샘플 개수: {len(forget10_dataset)}개")
        print(f"   └ 💾 로컬 파일 저장 완료: {forget_file_path}")
    except Exception as e:
        print(f"   └ ❌ 로드 및 저장 실패: {e}")
        forget10_dataset = []

    # 2. 🛡️ 보존해야 하는 90% 대조군 데이터셋 로드
    print("\n📦 [2/2] Retain 90% 데이터셋 처리 중...")
    try:
        raw_retain90 = load_dataset(path="locuslab/TOFU", name="retain90", split="train")
        retain90_dataset = [
            {
                "prompt": sample["question"].strip(),
                "response": sample["answer"].strip()
            }
            for sample in raw_retain90
        ]

        # 💾 로컬 폴더에 JSON 파일로 수납 저장
        retain_file_path = os.path.join(output_dir, "retain90_data.json")
        with open(retain_file_path, "w", encoding="utf-8") as f:
            json.dump(retain90_dataset, f, ensure_ascii=False, indent=2)

        print(f"   ├ ✅ 메모리 상주 완료! 샘플 개수: {len(retain90_dataset)}개")
        print(f"   └ 💾 로컬 파일 저장 완료: {retain_file_path}")
    except Exception as e:
        print(f"   └ ❌ 로드 및 저장 실패: {e}")
        retain90_dataset = []

    print("\n========================================================")
    print(f"🎉 모든 데이터 로컬 저장이 완료되었습니다! [위치: {os.path.abspath(output_dir)}]")
    print("========================================================")

    return forget10_dataset, retain90_dataset


if __name__ == "__main__":
    # 스크립트 실행
    forget_set, retain_set = load_and_save_tofu_datasets()

    # 🧪 파일 저장 후 데이터가 정상 가공되었는지 첫 샘플 가볍게 확인
    if forget_set:
        print("\n📝 [저장 데이터 샘플 확인]")
        print(f"   ├ Q (Prompt)  : {forget_set[0]['prompt']}")
        print(f"   └ A (Response): {forget_set[0]['response']}")