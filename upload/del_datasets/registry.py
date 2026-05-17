"""
레지스트리 관리 모듈
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


class RegistryManager:
    """레지스트리 관리"""

    def __init__(self, registry_path: str = 'registry.json'):
        self.registry_path = registry_path
        self.registry = self._load()

    def _load(self) -> Dict:
        """레지스트리 로드"""
        if Path(self.registry_path).exists():
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'datasets': [],
            'created': datetime.now().isoformat()
        }

    def save(self):
        """레지스트리 저장"""
        Path(self.registry_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)

    def add_entry(self, entry: Dict):
        """항목 추가"""
        self.registry['datasets'].append(entry)
        self.registry['last_updated'] = datetime.now().isoformat()
        self.save()

    def list_all(self) -> List[Dict]:
        """전체 목록"""
        return self.registry.get('datasets', [])

    def find(self, name: str, version: str = None) -> Dict:
        """특정 항목 찾기"""
        datasets = [d for d in self.registry['datasets'] if d['name'] == name]

        if not datasets:
            return None

        if version:
            for d in datasets:
                if d['version'] == version:
                    return d
            return None

        # 최신 버전
        return max(datasets, key=lambda x: x['version'])
