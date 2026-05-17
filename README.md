# unlearning-verify-project

LLM Model Privacy Protection

Data Validation & Registration Guide본 프로젝트는 UnlearningDatasetManager를 활용하여 데이터셋을 시스템에 등록하고, 등록 과정에서 자동화된 데이터 검증(Validation)을 수행합니다.이 가이드는 UnlearningDatasetManager를 사용하여 데이터셋을 올바르게 등록하고 검증하는 방법을 설명합니다.🚀 빠른 시작 (Quick Start)가장 기본적인 데이터 등록 및 검증 실행 방법은 다음과 같습니다. 프로젝트 루트 디렉토리에서 아래 예시 코드를 작성하거나 실행하여 데이터셋을 등록할 수 있습니다.실행 예시 (main.py)Pythonfrom upload.del_datasets.unlearning_manager import UnlearningDatasetManager

def main(): # 1. 언러닝 데이터셋 매니저 인스턴스 생성
manager = UnlearningDatasetManager()

    # 2. 데이터셋 등록 및 내부 검증 수행
    result = manager.register(
        dataset="example",
        name="Example Dataset",
        version="1.0"
    )

    print("Registration Result:", result)

if **name** == "**main**":
main()
🛠️ 주요 기능 및 검증 흐름manager.register() 메서드가 호출되면 내부적으로 다음과 같은 데이터 검증 및 등록 파이프라인이 작동합니다.파라미터 무결성 검사: dataset 식별자, name, version 등 필수 입력값이 누락되었거나 올바른 데이터 타입(String 등)인지 확인합니다.포맷 및 스키마 검증: 언러닝(Unlearning) 작업에 필요한 데이터셋의 규격, 컬럼 구조, 파일 포맷이 일치하는지 내부 검증 로직을 거칩니다.중복 및 버전 체크: 동일한 데이터셋 버전이 이미 등록되어 있는지 확인하여 데이터 오버라이트를 방지합니다.최종 등록: 검증이 성공적으로 완료되면 데이터셋 메타데이터를 시스템에 등록하고 result를 반환합니다.📋 API 레퍼런스UnlearningDatasetManager.register(...)데이터셋을 시스템에 검증 후 등록하는 핵심 메서드입니다.아규먼트 (Arguments)파라미터타입필수 여부설명datasetstrRequired데이터셋의 고유 식별자 키 (예: "example", "mnist_unlearn")namestrRequired사용자가 식별하기 쉬운 데이터셋의 공식 명칭versionstrRequired데이터셋의 버전 관리 형태 (예: "1.0", "2026-05-17")반환값 (Returns)result: 등록 성공 여부 상태 코드, 메타데이터 정보 또는 검증 결과 로그를 포함한 객체를 반환합니다.⚠️ 에러 핸들링 (Expected Errors)검증 과정에서 실패할 경우, UnlearningDatasetManager는 구체적인 예외(Exception)를 발생시킵니다. 대표적인 오류 유형은 다음과 같습니다.ValueError: 필수 파라미터가 누락되었거나 데이터 포맷이 올바르지 않은 경우.DuplicateDatasetError (예시): 이미 동일한 버전의 데이터셋이 등록되어 있는 경우.
