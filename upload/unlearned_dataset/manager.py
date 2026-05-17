"""통합 관리자"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from validator import DatasetValidator


class UnlearningDatasetManager:
    def __init__(self, registry_path: str = 'registry.json', output_dir: str = 'datasets'):
        self.registry_path = Path(registry_path)
        self.output_dir = Path(output_dir)
        self.validator = DatasetValidator()
        self._registry = self._load_registry()

    def _load_registry(self) -> Dict:
        if self.registry_path.exists():
            return json.loads(self.registry_path.read_text(encoding='utf-8'))
        return {'datasets': [], 'created': datetime.now().isoformat()}

    def _save_registry(self):
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(
            json.dumps(self._registry, indent=2, ensure_ascii=False), encoding='utf-8'
        )

    def register(self, dataset: List[Dict], name: str, version: str = "1.0",
                 description: str = "", force: bool = False) -> Dict[str, Any]:
        print(f"\n{'='*50}\n등록: {name} v{version}\n{'='*50}")
        result = self.validator.validate(dataset)
        self.validator.print_report(result)

        if not result['valid'] and not force:
            return {'success': False, 'message': 'Validation failed. Use force=True to override.', 'dataset_path': None}

        self.output_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.output_dir / f"{name.replace(' ', '_')}_v{version}.json"
        filepath.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding='utf-8')

        self._registry['datasets'].append({
            'name': name, 'version': version, 'description': description,
            'registered_at': datetime.now().isoformat(),
            'file_path': str(filepath),
            'stats': result['stats'],
            'validation_status': 'passed' if result['valid'] else 'warning'
        })
        self._registry['last_updated'] = datetime.now().isoformat()
        self._save_registry()

        print(f"✅ 등록 완료: {filepath}")
        return {'success': True, 'dataset_path': str(filepath)}

    def find(self, name: str, version: str = None) -> Dict:
        matches = [d for d in self._registry['datasets'] if d['name'] == name]
        if not matches:
            return None
        if version:
            return next((d for d in matches if d['version'] == version), None)
        return max(matches, key=lambda x: tuple(map(int, x['version'].split('.'))))

    def load(self, name: str, version: str = None) -> List[Dict]:
        info = self.find(name, version)
        return json.loads(Path(info['file_path']).read_text(encoding='utf-8')) if info else None

    def list_all(self) -> List[Dict]:
        return self._registry.get('datasets', [])


if __name__ == '__main__':
    manager = UnlearningDatasetManager()

    dataset = (
        [{'id': f'f{i:03d}', 'text': f'forget {i}', 'label': 'forget'} for i in range(5)] +
        [{'id': f'r{i:03d}', 'text': f'retain {i}', 'label': 'retain'} for i in range(10)]
    )

    result = manager.register(dataset, "Test", "1.0")
    if result['success']:
        print(f"사용: {result['dataset_path']}")
