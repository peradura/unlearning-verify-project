"""
통합 관리자 (모듈화 버전)
"""

from datetime import datetime
from typing import Dict, List, Any

from validator import DatasetValidator
from registry import RegistryManager
from file_manager import FileManager


class UnlearningDatasetManager:
    """검증 + 등록 통합 관리 (모듈화)"""

    def __init__(self, registry_path: str = 'registry.json', output_dir: str = 'datasets'):
        self.validator = DatasetValidator()
        self.registry = RegistryManager(registry_path)
        self.file_manager = FileManager(output_dir)

    def validate(self, dataset: List[Dict], verbose: bool = True) -> Dict[str, Any]:
        """검증만 수행"""
        result = self.validator.validate(dataset)

        if verbose:
            self.validator.print_report(result)

        return result

    def register(self,
                dataset: List[Dict],
                name: str,
                version: str = "1.0",
                description: str = "",
                force: bool = False) -> Dict[str, Any]:
        """검증 + 등록"""

        print("=" * 70)
        print(f"등록: {name} v{version}")
        print("=" * 70)

        # 1. 검증
        validation = self.validate(dataset, verbose=True)

        if not validation['valid'] and not force:
            return {
                'success': False,
                'message': 'Validation failed. Use force=True to override.',
                'dataset_path': None,
                'validation': validation
            }

        # 2. 파일 저장
        filepath = self.file_manager.save_dataset(dataset, name, version)

        # 3. 레지스트리 업데이트
        entry = {
            'name': name,
            'version': version,
            'description': description,
            'registered_at': datetime.now().isoformat(),
            'file_path': filepath,
            'stats': validation['stats'],
            'validation_status': 'passed' if validation['valid'] else 'warning'
        }

        self.registry.add_entry(entry)

        print(f"\n✅ 등록 완료: {filepath}")
        print("=" * 70)

        return {
            'success': True,
            'message': f'Registered: {name} v{version}',
            'dataset_path': filepath,
            'validation': validation
        }

    def list_datasets(self) -> List[Dict]:
        """등록된 데이터셋 목록"""
        return self.registry.list_all()

    def get_dataset(self, name: str, version: str = None) -> Dict:
        """특정 데이터셋 조회"""
        return self.registry.find(name, version)

    def load_dataset(self, name: str, version: str = None) -> List[Dict]:
        """데이터셋 로드"""
        info = self.get_dataset(name, version)
        if info:
            return self.file_manager.load_dataset(info['file_path'])
        return None


# 사용 예제
if __name__ == '__main__':
    # 초기화
    manager = UnlearningDatasetManager()

    # 예제 데이터
    dataset = [
        {'id': f'f{i:03d}', 'text': f'forget {i}', 'label': 'forget'}
        for i in range(5)
    ] + [
        {'id': f'r{i:03d}', 'text': f'retain {i}', 'label': 'retain'}
        for i in range(6)
    ]

    # 등록
    result = manager.register(dataset, "Test", "1.0")

    if result['success']:
        print(f"\n사용: {result['dataset_path']}")
