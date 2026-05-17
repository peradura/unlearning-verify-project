"""
검증 전용 모듈
"""

from typing import Dict, List, Any


class DatasetValidator:
    """데이터셋 검증"""

    REQUIRED_FIELDS = ['id', 'text', 'label']
    VALID_LABELS = ['forget', 'retain', 'validation']
    MIN_SAMPLES = 10

    def validate(self, dataset: List[Dict]) -> Dict[str, Any]:
        """검증 수행"""
        errors = []
        warnings = []

        # 1. 기본 체크
        if not isinstance(dataset, list) or len(dataset) == 0:
            return {
                'valid': False,
                'errors': ['Dataset is empty or not a list'],
                'warnings': [],
                'stats': {}
            }

        # 2. 필드 및 라벨 검증
        label_counts = {'forget': 0, 'retain': 0, 'validation': 0}

        for idx, item in enumerate(dataset):
            # 필수 필드
            missing = [f for f in self.REQUIRED_FIELDS if f not in item]
            if missing:
                errors.append(f"Item {idx}: Missing {missing}")
                continue

            # 라벨 검증
            label = item['label']
            if label not in self.VALID_LABELS:
                errors.append(f"Item {idx}: Invalid label '{label}'")
            else:
                label_counts[label] += 1

            # 텍스트 검증
            if not isinstance(item['text'], str) or len(item['text'].strip()) == 0:
                warnings.append(f"Item {idx}: Empty or invalid text")

        # 3. 분포 검증
        if label_counts['forget'] == 0:
            errors.append("No 'forget' samples")

        if len(dataset) < self.MIN_SAMPLES:
            errors.append(f"Insufficient samples: {len(dataset)} (min: {self.MIN_SAMPLES})")

        if label_counts['retain'] < label_counts['forget']:
            warnings.append("retain < forget (catastrophic forgetting risk)")

        # 4. 중복 검증
        ids = [item.get('id') for item in dataset]
        duplicates = [x for x in ids if ids.count(x) > 1]
        if duplicates:
            warnings.append(f"Duplicate IDs: {set(duplicates)}")

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'stats': {
                'total': len(dataset),
                'forget': label_counts['forget'],
                'retain': label_counts['retain'],
                'validation': label_counts['validation']
            }
        }

    def print_report(self, result: Dict):
        """검증 리포트 출력"""
        print("=" * 70)
        print("검증 리포트")
        print("=" * 70)

        status = "✅ 통과" if result['valid'] else "❌ 실패"
        print(f"\n상태: {status}")

        stats = result['stats']
        if stats:
            print(f"\n통계:")
            print(f"  전체: {stats['total']}개")
            print(f"  forget: {stats['forget']}개")
            print(f"  retain: {stats['retain']}개")

        if result['errors']:
            print(f"\n오류 ({len(result['errors'])}개):")
            for err in result['errors']:
                print(f"  • {err}")

        if result['warnings']:
            print(f"\n경고 ({len(result['warnings'])}개):")
            for warn in result['warnings']:
                print(f"  • {warn}")

        print("=" * 70)
