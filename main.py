# from upload.del_datasets.manager import UnlearningDatasetManager

# def main():
    
#     manager = UnlearningDatasetManager()
#     result = manager.register(
#         dataset="example",
#         name="Example Dataset",
#         version="1.0"
#     )

# if __name__ == "__main__":
#     main()


import os
from datetime import datetime
from upload.del_datasets.manager import UnlearningDatasetManager
from upload.del_datasets.validator import DatasetValidator

def main():
    print("🚀 Unlearning Dataset Manager 테스트 시작...")
    
    # 2. 매니저 초기화 
    # 프로젝트 루트에 test_output 폴더를 만들어 데이터를 깔끔하게 관리하도록 경로 설정
    registry_path = 'upload/del_datasets/saved/test_registry.json'
    output_dir = 'upload/del_datasets/saved/test_datasets'
    
    manager = UnlearningDatasetManager(
        registry_path=registry_path, 
        output_dir=output_dir
    )

    # ------------------------------------------------------------
    # Case 1: 모든 검증 조건을 만족하는 정상 데이터셋
    # ------------------------------------------------------------
    print("\n[테스트 1] 정상 데이터셋 등록 시도")
    success_dataset = [
        {"id": f"f_{i:03d}", "text": f"Target forget sample text {i}", "label": "forget"} 
        for i in range(5)
    ] + [
        {"id": f"r_{i:03d}", "text": f"General retain knowledge text {i}", "label": "retain"} 
        for i in range(12)
    ]

    res1 = manager.register(
        success_dataset, 
        name="Toy Unlearning Spec", 
        version="1.0", 
        description="정상 작동 확인용 토이 데이터셋"
    )

    # ------------------------------------------------------------
    # Case 2: 경고(Warning)가 발생하는 데이터셋 
    # (총 개수 10개는 채웠으나 retain < forget 관계 및 공백 텍스트 포함)
    # ------------------------------------------------------------
    print("\n[테스트 2] 경고(Warning) 발생 데이터셋 등록 시도")
    warning_dataset = [
        {"id": "w_01", "text": "Forget this core entity info", "label": "forget"},
        {"id": "w_02", "text": "Forget this too", "label": "forget"},
        {"id": "w_03", "text": "   ", "label": "forget"}, # ⚠️ 공백 텍스트 경고 유도
        {"id": "w_04", "text": "General retain stream", "label": "retain"}, # ⚠️ retain(1) < forget(3) 경고 유도
    ] + [
        {"id": f"w_val_{i}", "text": f"Validation split anchor {i}", "label": "validation"} 
        for i in range(7) # 총 11개로 최소 샘플 개수(10개)는 충족
    ]

    res2 = manager.register(warning_dataset, name="Warning Spec Test", version="1.0")

    # ------------------------------------------------------------
    # Case 3: 에러(Error)로 인해 등록이 완전히 실패하는 데이터셋
    # ------------------------------------------------------------
    print("\n[테스트 3] 에러(Error) 데이터셋 등록 시도 (실패 예상)")
    broken_dataset = [
        {"id": "e_01", "text": "Valid retain but...", "label": "retain"},
        {"id": "e_02", "label": "retain"}, # ❌ text 필드 누락 오류
        {"id": "e_03", "text": "Invalid label mapping", "label": "wrong_label"}, # ❌ 정의되지 않은 라벨 오류
    ] # ❌ 최소 샘플 수(10개) 미달, forget 샘플 0개 오류

    res3 = manager.register(broken_dataset, name="Broken Spec Test", version="1.0")

    # ------------------------------------------------------------
    # Case 4: 등록된 데이터셋 파일 조회 및 로드 테스트
    # ------------------------------------------------------------
    print("\n[테스트 4] Registry 파일 연동 및 로드 테스트")
    
    all_stored = manager.list_all()
    print(f"📦 현재 Registry에 등록된 세션 수: {len(all_stored)}개")
    
    if res1['success']:
        print("\n'Toy Unlearning Spec' 데이터셋을 다시 가져옵니다...")
        loaded_data = manager.load("Toy Unlearning Spec", version="1.0")
        if loaded_data:
            print(f"✅ 파일 로드 성공! 총 로드된 샘플 수: {len(loaded_data)}개")
            print(f"🔍 첫 번째 샘플 구조: {loaded_data[0]}")

if __name__ == '__main__':
    main()