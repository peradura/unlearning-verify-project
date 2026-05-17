from upload.unlearned_dataset.manager import UnlearningDatasetManager

def main():    
    # dataset 검증 / 등록
    manager = UnlearningDatasetManager()
    result = manager.register(
        dataset="example",
        name="Example Dataset",
        version="1.0"
    )
    
    # 경로 수정 필요
    forget_dataset_path = result.get("forget_path", "path/to/forget_set.jsonl")
    nonmember_dataset_path = result.get("nonmember_path", "path/to/non_member_set.jsonl")
    
    # 평가 대상이 되는 Unlearned 모델 경로, 경로 수정 필요
    target_model_path = "/path/to/your/unlearned_model"
    
    # l2 distance 평가
    config_data = "bert-base-uncased"  # 본인의 모델에 맞게 변경하세요.
    
    # 2. 오리지널 모델과 변경(언러닝) 모델의 가중치 파일(.pt 또는 .bin) 경로
    orig_weights_path = "./weights/original_model.pt"
    unlearn_weights_path = "./weights/unlearned_model.pt"
    
    # 3. 디바이스 설정 (CUDA 사용이 불가능하면 cpu로 변경)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 가상의 데이터셋과 토크나이저는 현재 함수 내부에서 주석 처리되어 있으므로, 
    # None으로 넘겨주어도 정상 작동
    unlearn_dataset = None
    tokenizer = None

    print("레이어별 가중치 L2 Distance 계산 시작...")
    try:
        # 함수 호출
        weight_l2_distances = calculate_metrics(
            config_data=config_data,
            orig_weights=orig_weights_path,
            unlearn_weights=unlearn_weights_path,
            unlearn_dataset=unlearn_dataset,
            tokenizer=tokenizer,
            device=device
        )
        
        # 결과 출력 (상위 10개 레이어만 예시로 출력)
        print("\n=== 계산 완료 (Normalized L2 Distance) ===")
        for i, (layer_name, dist) in enumerate(weight_l2_distances.items()):
            print(f"{layer_name}: {dist:.6f}")
            if i >= 15: # 너무 많으면 끊어서 보기
                print("...")
                break
                
    except Exception as e:
        print(f"실행 중 오류가 발생했습니다: {e}")
    
    # MIA 평가
    mia_config = MIAConfig(
        model_path=target_model_path,
        forget_data=forget_dataset_path,
        nonmember_data=nonmember_dataset_path,
        output_path="mia_results.json"
    )
    
    # 주입된 설정을 바탕으로 MIA 평가 함수 실행
    run_evaluation(mia_config)

if __name__ == "__main__":
    main()
    