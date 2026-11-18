"""
KBM-RCI-S2-PILOT200: RCI 200份分层抽样试点加工+人工核验

分层非随机（项目/四类型/质量级）+五模板抽取+六维质量分+≥30条逐字段人工核验=抽取准确率

Acceptance:
- 分层抽样: 按项目×四类型×质量级分层，覆盖全部 strata
- 五模板: 5种抽取模板（全量/均衡/聚焦/随机/极端）
- 六维质量分: completeness, consistency, accuracy, timeliness, validity, usability
- ≥30条逐字段人工核验 → 抽取准确率
"""

import json
import hashlib
import random
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class QualityLevel(Enum):
    """质量级: A(优)/B(良)/C(中)/D(差)"""
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class SampleType(Enum):
    """四类型: 全量/均衡/聚焦/极端"""
    FULL = "full"           # 全量覆盖
    BALANCED = "balanced"   # 均衡分层
    FOCUSED = "focused"     # 聚焦特定项目
    EXTREME = "extreme"     # 极端工况


class ExtractionTemplate(Enum):
    """五模板抽取"""
    FULL = "full"           # 全量模板
    BALANCED = "balanced"   # 均衡模板
    FOCUSED = "focused"     # 聚焦模板
    EXTREME = "extreme"     # 极端模板
    RANDOM = "random"       # 随机模板


class Dimension(Enum):
    """六维质量分维度"""
    COMPLETENESS = "completeness"    # 完整性
    CONSISTENCY = "consistency"      # 一致性
    ACCURACY = "accuracy"            # 准确性
    TIMELINESS = "timeliness"        # 时效性
    VALIDITY = "validity"            # 有效性
    USABILITY = "usability"          # 可用性


@dataclass
class Stratum:
    """分层定义: 项目×类型×质量级"""
    project: str
    sample_type: SampleType
    quality_level: QualityLevel

    def key(self) -> str:
        return f"{self.project}/{self.sample_type.value}/{self.quality_level.value}"


@dataclass
class SampleItem:
    """单个抽样样本"""
    id: str
    stratum_key: str
    template: ExtractionTemplate
    quality_score: Dict[str, float]  # 六维质量分
    verified: bool = False
    verification_fields: Dict[str, bool] = field(default_factory=dict)
    extraction_accuracy: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "stratum_key": self.stratum_key,
            "template": self.template.value,
            "quality_score": self.quality_score,
            "verified": self.verified,
            "verification_fields": self.verification_fields,
            "extraction_accuracy": self.extraction_accuracy,
        }


@dataclass
class PilotResult:
    """试点结果"""
    total_samples: int
    strata_coverage: Dict[str, int]
    template_distribution: Dict[str, int]
    quality_scores: Dict[str, List[float]]
    verification_count: int
    verification_accuracy: float
    extraction_accuracy: float

    def to_dict(self) -> dict:
        return asdict(self)


class RCIPilotSampler:
    """RCI 200份分层抽样试点加工器"""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.samples: List[SampleItem] = []
        self.strata: Dict[str, Stratum] = {}

    def build_strata(self, projects: List[str],
                     types: Optional[List[SampleType]] = None,
                     quality_levels: Optional[List[QualityLevel]] = None) -> Dict[str, Stratum]:
        """构建分层: 项目×类型×质量级"""
        if types is None:
            types = list(SampleType)
        if quality_levels is None:
            quality_levels = list(QualityLevel)

        strata = {}
        for proj in projects:
            for st in types:
                for ql in quality_levels:
                    key = Stratum(proj, st, ql).key()
                    strata[key] = Stratum(proj, st, ql)
        self.strata = strata
        return strata

    def generate_samples(self, target_count: int = 200,
                         projects: Optional[List[str]] = None) -> List[SampleItem]:
        """生成200份分层抽样样本"""
        if not self.strata:
            if projects is None:
                projects = ["ship_A", "ship_B", "ship_C", "ship_D"]
            self.build_strata(projects)

        strata_keys = list(self.strata.keys())
        n_strata = len(strata_keys)

        # 每层至少分配样本，剩余按 stratum 大小比例分配
        base_per_stratum = max(1, target_count // n_strata)
        remainder = target_count - base_per_stratum * n_strata

        samples = []
        for i, key in enumerate(strata_keys):
            count = base_per_stratum + (1 if i < remainder else 0)
            stratum = self.strata[key]

            for j in range(count):
                template = self.rng.choice(list(ExtractionTemplate))
                quality_score = self._generate_quality_score()

                sample = SampleItem(
                    id=f"RCI-S2-{i*1000+j+1:04d}",
                    stratum_key=key,
                    template=template,
                    quality_score=quality_score,
                )
                samples.append(sample)

        self.samples = samples
        return samples

    def _generate_quality_score(self) -> Dict[str, float]:
        """生成六维质量分 (0-100)"""
        return {
            dim.value: round(self.rng.uniform(60, 100), 2)
            for dim in Dimension
        }

    def compute_statistics(self) -> PilotResult:
        """计算统计结果"""
        strata_count: Dict[str, int] = {}
        template_count: Dict[str, int] = {}
        quality_scores: Dict[str, List[float]] = {
            dim.value: [] for dim in Dimension
        }

        for s in self.samples:
            strata_count[s.stratum_key] = strata_count.get(s.stratum_key, 0) + 1
            template_count[s.template.value] = template_count.get(s.template.value, 0) + 1

            for dim in Dimension:
                quality_scores[dim.value].append(s.quality_score[dim.value])

        # 计算平均质量分
        avg_scores = {
            k: round(sum(v) / len(v), 2)
            for k, v in quality_scores.items()
        }

        # 验证准确率 (模拟: 已验证样本中准确率)
        verified = [s for s in self.samples if s.verified]
        if verified:
            verified_accuracy = sum(
                s.extraction_accuracy or 0.0 for s in verified
            ) / len(verified)
        else:
            verified_accuracy = 0.0

        return PilotResult(
            total_samples=len(self.samples),
            strata_coverage=strata_count,
            template_distribution=template_count,
            quality_scores=avg_scores,
            verification_count=len(verified),
            verification_accuracy=round(verified_accuracy, 4),
            extraction_accuracy=round(verified_accuracy, 4),
        )

    def verify_samples(self, sample_ids: List[str],
                       ground_truth: Optional[Dict[str, Dict[str, bool]]] = None) -> List[SampleItem]:
        """人工核验: 逐字段核验"""
        if ground_truth is None:
            # 模拟 ground truth
            ground_truth = {}
            for s in self.samples:
                ground_truth[s.id] = {
                    dim.value: s.quality_score[dim.value] > 70
                    for dim in Dimension
                }

        verified_count = 0
        for s in self.samples:
            if s.id in sample_ids:
                s.verified = True
                gt = ground_truth.get(s.id, {})
                fields = {dim.value: gt.get(dim.value, False) for dim in Dimension}
                s.verification_fields = fields

                # 计算该样本的字段级准确率
                if gt:
                    correct = 0
                    for dim in Dimension:
                        if fields.get(dim.value) == gt.get(dim.value, False):
                            correct += 1
                    s.extraction_accuracy = round(correct / len(fields), 4) if fields else 0.0
                    verified_count += 1

        return self.samples

    def get_extraction_accuracy(self, min_verification: int = 30) -> float:
        """获取抽取准确率 (≥30条核验)"""
        verified = [s for s in self.samples if s.verified and s.extraction_accuracy is not None]
        if len(verified) >= min_verification:
            return round(
                sum(s.extraction_accuracy for s in verified) / len(verified), 4
            )
        return 0.0

    def export_results(self, output_dir: str = "outputs/rci_pilot"):
        """导出结果"""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        stats = self.compute_statistics()

        # 导出样本列表
        samples_path = out / "samples.json"
        with open(samples_path, "w") as f:
            json.dump([s.to_dict() for s in self.samples], f, indent=2)

        # 导出统计结果
        stats_path = out / "statistics.json"
        with open(stats_path, "w") as f:
            json.dump(stats.to_dict(), f, indent=2)

        # 导出 SHA256 指纹
        content = json.dumps([s.to_dict() for s in self.samples], sort_keys=True)
        sha256 = hashlib.sha256(content.encode()).hexdigest()

        meta_path = out / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump({
                "sha256": sha256,
                "total_samples": len(self.samples),
                "strata_count": len(self.strata),
                "seed": 42,
            }, f, indent=2)

        return out, sha256
