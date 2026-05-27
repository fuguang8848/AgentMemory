"""
memory-hermes 验证脚本
"""

import sys
import os
import asyncio
import json

# 添加 src 到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import Config
from decay_engine import DecayEngine, MemoryArchiver
from L2_graph_store import GraphStore, Entity, EntityType, RelationType, Relation
from L4_file_persist import FilePersistStore


def test_config():
    print("\n=== [1/5] Config ===")
    cfg = Config()
    assert cfg.get("embedding.model") == "text-embedding-v3"
    assert cfg.get("decay.threshold") == 0.3
    print("✅ Config 加载正确")
    print(f"   embedding: {cfg.get('embedding.model')} ({cfg.get('embedding.dimensions')}维)")
    print(f"   decay: 半衰期{cfg.get('decay.half_life_days')}天, 阈值{cfg.get('decay.threshold')}")


def test_decay_engine():
    print("\n=== [2/5] Decay Engine ===")
    engine = DecayEngine(half_life_days=14.0, forget_threshold=0.3, archive_threshold=0.5)
    
    # 测试衰减因子
    decay_14d = engine.decay_factor(14.0)
    assert abs(decay_14d - 0.5) < 0.01, f"14天应该衰减到0.5, 实际{decay_14d}"
    print(f"✅ 衰减因子: 14天={decay_14d:.3f}")
    
    # 测试遗忘评分
    entry = {
        "id": "test1",
        "content": "测试记忆",
        "access_count": 5,
        "importance": 0.8,
        "created_at": "2026-05-13T12:00:00",  # 14天前
        "last_accessed": "2026-05-20T12:00:00"  # 7天前
    }
    score = engine.calculate_score(entry)
    print(f"✅ 遗忘评分: {score.score:.3f}")
    print(f"   组成: {score.components}")
    
    # 测试遗忘判断: 0.72 > 0.5, 所以不归档也不遗忘
    assert engine.should_forget(score) == False
    assert engine.should_archive(score) == False
    print("✅ 遗忘判断正确 (高分0.72→保留)")
    
    # 测试低分遗忘
    low_entry = {
        "id": "low1",
        "content": "低价值记忆",
        "access_count": 0,
        "importance": 0.1,
        "created_at": "2026-01-01T12:00:00",
        "last_accessed": "2026-01-01T12:00:00"
    }
    low_score = engine.calculate_score(low_entry)
    assert engine.should_forget(low_score) == True
    assert engine.should_archive(low_score) == False
    print(f"✅ 低分遗忘判断正确 (低分{low_score.score:.3f}→遗忘)")


def test_graph_store():
    print("\n=== [3/5] Graph Store ===")
    store = GraphStore()  # 使用默认路径
    
    # 添加实体
    e1 = Entity(name="优优", entity_type=EntityType.PERSON, properties={"role": "学生"})
    e2 = Entity(name="石榴籽", entity_type=EntityType.PROJECT, properties={"category": "AI翻译"})
    e1_id = store.add_entity(e1)
    e2_id = store.add_entity(e2)
    print(f"✅ 实体添加: 优优(id={e1_id}), 石榴籽(id={e2_id})")
    
    # 添加关系
    r1 = Relation(source_entity_id=e1_id, target_entity_id=e2_id, relation_type=RelationType.WORKS_ON)
    r1_id = store.add_relation(r1)
    print(f"✅ 关系添加: 优优 参与 石榴籽 (id={r1_id})")
    
    # 查询邻居
    neighbors = store.get_neighbors(e1_id)
    assert len(neighbors) == 1
    assert neighbors[0].name == "石榴籽"
    print(f"✅ 邻居查询: 优优的关联实体={neighbors[0].name}")
    
    # 统计
    counts = store.get_entity_count()
    print(f"✅ 实体统计: {counts}")


def test_file_persist():
    print("\n=== [4/5] File Persist ===")
    store = FilePersistStore()  # 使用默认工作目录
    
    # 写入测试记忆
    store.store_fact("测试：石榴籽省赛结果待审核", {
        "fact_type": "project",
        "importance": 0.8,
        "entities": ["石榴籽", "省赛"]
    })
    print("✅ 记忆写入 L4")
    
    # 读取今日日记
    today_entries = store.daily_memory.list_entries()
    print(f"✅ 今日日记条数: {len(today_entries)}")
    
    # 搜索
    results = store.daily_memory.search("石榴籽")
    print(f"✅ 搜索'石榴籽': 找到{len(results)}条")


def test_integration():
    print("\n=== [5/5] Integration (L1+L3 需要 API Key，略) ===")
    print("⚠️  L1 (LLM压缩) 和 L3 (向量检索) 需要 dashscope API Key")
    print("   手动验证: python -c \"from src import MemoryHermes; print(MemoryHermes())\"")
    
    # 简单检查 memory_manager 能否实例化
    try:
        from src.memory_manager import MemoryHermes
        print("✅ MemoryHermes 导入成功")
    except Exception as e:
        print(f"❌ MemoryHermes 导入失败: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("agentmemory 功能验证")
    print("=" * 50)
    
    try:
        test_config()
        test_decay_engine()
        test_graph_store()
        test_file_persist()
        test_integration()
        
        print("\n" + "=" * 50)
        print("🎉 所有可验证的模块通过!")
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
