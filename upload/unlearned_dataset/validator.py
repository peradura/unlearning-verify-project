"""검증 모듈"""

from collections import Counter
from typing import Dict, List, Any


class DatasetValidator:
    REQUIRED_FIELDS = ['id', 'text', 'label']
    VALID_LABELS = ['forget', 'retain', 'validation']
    MIN_SAMPLES = 10

    def validate(self, dataset: List[Dict]) -> Dict[str, Any]:
        if not isinstance(dataset, list) or not dataset:
            return {'valid': False, 'errors': ['Dataset is empty or not a list'], 'warnings': [], 'stats': {}}

        errors, warnings = [], []
        counts = {'forget': 0, 'retain': 0, 'validation': 0}

        for i, item in enumerate(dataset):
            missing = [f for f in self.REQUIRED_FIELDS if f not in item]
            if missing:
                errors.append(f"Item {i}: Missing {missing}")
                continue

            if item['label'] not in self.VALID_LABELS:
                errors.append(f"Item {i}: Invalid label '{item['label']}'")
            else:
                counts[item['label']] += 1

            if not isinstance(item['text'], str) or not item['text'].strip():
                warnings.append(f"Item {i}: Empty text")

        if not counts['forget']:
            errors.append("No 'forget' samples")
        if len(dataset) < self.MIN_SAMPLES:
            errors.append(f"Too few samples: {len(dataset)} (min: {self.MIN_SAMPLES})")
        if counts['retain'] < counts['forget']:
            warnings.append("retain < forget (catastrophic forgetting risk)")

        dupes = [id_ for id_, c in Counter(
            item['id'] for item in dataset if 'id' in item
        ).items() if c > 1]
        if dupes:
            warnings.append(f"Duplicate IDs: {set(dupes)}")

        return {
            'valid': not errors,
            'errors': errors,
            'warnings': warnings,
            'stats': {'total': len(dataset), **counts}
        }

    def print_report(self, result: Dict):
        sep = "=" * 50
        print(f"{sep}\n{'✅ 통과' if result['valid'] else '❌ 실패'}")
        if s := result['stats']:
            print(f"전체: {s['total']} | forget: {s['forget']} | retain: {s['retain']} | validation: {s['validation']}")
        for e in result['errors']:
            print(f"  오류: {e}")
        for w in result['warnings']:
            print(f"  경고: {w}")
        print(sep)
