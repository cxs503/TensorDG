"""
KBM-RCI-S2-PILOT200 测试: 分层抽样试点加工+人工核验

验收标准:
1. 分层覆盖: 200份样本覆盖全部 strata (项目×类型×质量级)
2. 五模板分布: 5种抽取模板均有分配
3. 六维质量分: 每个样本有 completeness/consistency/accuracy/timeliness/validity/usability
4. ≥30条逐字段人工核验 → 抽取准确率
5. SHA256 指纹: 确定性输出
"""

import json
import hashlib
import pytest
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tensordg.rci_pilot_sampler import (
    RCIPilotSampler,
    QualityLevel,
    SampleType,
    ExtractionTemplate,
    Dimension,
    Stratum,
    SampleItem,
    PilotResult,
)


class TestStratum:
    """分层定义测试"""

    def test_stratum_key(self):
        s = Stratum("ship_A", SampleType.FULL, QualityLevel.A)
        assert s.key() == "ship_A/full/A"

    def test_stratum_equality(self):
        s1 = Stratum("ship_A", SampleType.FULL, QualityLevel.A)
        s2 = Stratum("ship_A", SampleType.FULL, QualityLevel.A)
        assert s1.key() == s2.key()


class TestRCIPilotSampler:
    """RCI 分层抽样试点加工器测试"""

    def test_build_strata(self):
        """测试分层构建"""
        sampler = RCIPilotSampler(seed=42)
        projects = ["ship_A", "ship_B"]
        strata = sampler.build_strata(projects)

        # 2 项目 × 4 类型 × 4 质量级 = 32 strata
        assert len(strata) == 32

        # 检查 key 格式
        for key in strata:
            parts = key.split("/")
            assert len(parts) == 3
            assert parts[0] in projects
            assert parts[1] in [t.value for t in SampleType]
            assert parts[2] in [q.value for q in QualityLevel]

    def test_generate_200_samples(self):
        """测试生成200份样本"""
        sampler = RCIPilotSampler(seed=42)
        projects = ["ship_A", "ship_B", "ship_C", "ship_D"]
        samples = sampler.generate_samples(target_count=200, projects=projects)

        assert len(samples) == 200

        # 检查样本 ID 格式
        for s in samples:
            assert s.id.startswith("RCI-S2-")
            assert s.stratum_key in sampler.strata

    def test_template_distribution(self):
        """测试五模板分布"""
        sampler = RCIPilotSampler(seed=42)
        projects = ["ship_A", "ship_B", "ship_C", "ship_D"]
        samples = sampler.generate_samples(target_count=200, projects=projects)

        template_counts = {}
        for s in samples:
            template_counts[s.template.value] = template_counts.get(s.template.value, 0)

        # 五模板均有分配
        assert len(template_counts) == 5
        for template in ExtractionTemplate:
            assert template.value in template_counts

    def test_quality_scores(self):
        """测试六维质量分"""
        sampler = RCIPilotSampler(seed=42)
        projects = ["ship_A"]
        samples = sampler.generate_samples(target_count=10, projects=projects)

        for s in samples:
            for dim in Dimension:
                assert dim.value in s.quality_score
                assert 0 <= s.quality_score[dim.value] <= 100

    def test_verification(self):
        """测试人工核验 (≥30条)"""
        sampler = RCIPilotSampler(seed=42)
        projects = ["ship_A", "ship_B", "ship_C", "ship_D"]
        samples = sampler.generate_samples(target_count=200, projects=projects)

        # 选取 30+ 条进行核验
        verify_ids = [s.id for s in samples[:35]]
        sampler.verify_samples(verify_ids)

        verified = [s for s in samples if s.verified]
        assert len(verified) == 35

        # 检查核验字段
        for s in verified:
            assert len(s.verification_fields) == 6  # 六维

    def test_extraction_accuracy(self):
        """测试抽取准确率计算"""
        sampler = RCIPilotSampler(seed=42)
        projects = ["ship_A", "ship_B", "ship_C", "ship_D"]
        samples = sampler.generate_samples(target_count=200, projects=projects)

        # 核验 30+ 条
        verify_ids = [s.id for s in samples[:35]]
        sampler.verify_samples(verify_ids)

        accuracy = sampler.get_extraction_accuracy(min_verification=30)
        assert accuracy >= 0.0
        assert accuracy <= 1.0

    def test_statistics(self):
        """测试统计结果"""
        sampler = RCIPilotSampler(seed=42)
        projects = ["ship_A", "ship_B"]
        samples = sampler.generate_samples(target_count=100, projects=projects)

        stats = sampler.compute_statistics()

        assert stats.total_samples == 100
        assert len(stats.strata_coverage) > 0
        assert len(stats.template_distribution) == 5
        assert len(stats.quality_scores) == 6  # 六维

    def test_export_results(self):
        """测试结果导出"""
        sampler = RCIPilotSampler(seed=42)
        projects = ["ship_A", "ship_B"]
        samples = sampler.generate_samples(target_count=50, projects=projects)

        out_dir, sha256 = sampler.export_results("outputs/rci_pilot_test")

        assert Path(out_dir).exists()
        assert Path(out_dir / "samples.json").exists()
        assert Path(out_dir / "statistics.json").exists()
        assert Path(out_dir / "metadata.json").exists()
        assert len(sha256) == 64  # SHA256 hex

    def test_deterministic_output(self):
        """测试确定性输出 (SHA256 一致)"""
        sampler1 = RCIPilotSampler(seed=42)
        sampler2 = RCIPilotSampler(seed=42)

        projects = ["ship_A", "ship_B"]
        samples1 = sampler1.generate_samples(target_count=50, projects=projects)
        samples2 = sampler2.generate_samples(target_count=50, projects=projects)

        content1 = json.dumps([s.to_dict() for s in samples1], sort_keys=True)
        content2 = json.dumps([s.to_dict() for s in samples2], sort_keys=True)

        assert content1 == content2
        assert hashlib.sha256(content1.encode()).hexdigest() == hashlib.sha256(content2.encode()).hexdigest()

    def test_strata_coverage(self):
        """测试分层覆盖"""
        sampler = RCIPilotSampler(seed=42)
        projects = ["ship_A", "ship_B", "ship_C"]
        samples = sampler.generate_samples(target_count=200, projects=projects)

        # 3 项目 × 4 类型 × 4 质量级 = 48 strata
        expected_strata = 48
        covered = set(s.stratum_key for s in samples)
        assert len(covered) == expected_strata

    def test_quality_score_range(self):
        """测试质量分范围"""
        sampler = RCIPilotSampler(seed=42)
        projects = ["ship_A"]
        samples = sampler.generate_samples(target_count=100, projects=projects)

        for s in samples:
            for dim in Dimension:
                score = s.quality_score[dim.value]
                assert 60 <= score <= 100, f"Quality score {score} out of range for {dim.value}"


class TestIntegration:
    """集成测试: 完整流程"""

    def test_full_pipeline(self):
        """完整流程: 分层→抽样→核验→统计"""
        sampler = RCIPilotSampler(seed=42)
        projects = ["ship_A", "ship_B", "ship_C", "ship_D"]

        # 1. 构建分层
        strata = sampler.build_strata(projects)
        assert len(strata) == 4 * 4 * 4  # 4×4×4 = 64

        # 2. 生成200份样本
        samples = sampler.generate_samples(target_count=200, projects=projects)
        assert len(samples) == 200

        # 3. 人工核验 ≥30 条
        verify_ids = [s.id for s in samples[:35]]
        sampler.verify_samples(verify_ids)

        # 4. 计算统计
        stats = sampler.compute_statistics()
        assert stats.verification_count == 35
        assert stats.extraction_accuracy >= 0.0

        # 5. 导出
        out_dir, sha256 = sampler.export_results("outputs/rci_pilot_integration")
        assert Path(out_dir).exists()
        assert len(sha256) == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
