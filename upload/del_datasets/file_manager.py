"""
파일 저장/로드 모듈
"""

import json
from pathlib import Path
from typing import List, Dict


class FileManager:
    """파일 관리"""

    def __init__(self, output_dir: str = 'datasets'):
        self.output_dir = output_dir

    def save_dataset(self, dataset: List[Dict], name: str, version: str) -> str:
        """데이터셋 저장"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        filename = f"{name.replace(' ', '_')}_v{version}.json"
        filepath = Path(self.output_dir) / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        return str(filepath)

    def load_dataset(self, filepath: str) -> List[Dict]:
        """데이터셋 로드"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
