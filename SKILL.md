---
name: agentmemory
version: 1.0.0
family: symphony
role: memory-center
description: 交响乐技能家族 - 顶尖记忆系统，融合 Hermes + Mem0 优点，四层闭环记忆架构
---

# agentmemory 顶尖记忆技能

> 交响乐技能家族成员 | 整合 Hermes 记忆机制 + Mem0 混合检索 + 四层闭环架构

## 核心能力

| 能力 | 说明 |
|------|------|
| `memory.store` | 存储记忆（自动 LLM 事实提取） |
| `memory.query` | 查询记忆（混合检索：向量+BM25+重要性） |
| `memory.prefetch` | 预取相关记忆（后台异步加载） |
| `memory.forget` | 主动遗忘（评分驱动） |
| `memory.explain_rank` | 解释排名理由 |
| `memory.compact` | 记忆压缩（合并相似记忆） |
| `memory.stats` | 记忆统计 |

## 四层闭环架构

```
L1: LCM 压缩层（对话 → 关键事实）
    └── LLM 提取事实，不存原始对话

L2: Graph 图谱层（事实 → 实体关系）
    └── 实体（人名/项目）+ 关系 + 属性

L3: Vector 向量层（混合检索）
    └── 向量语义(60%) + BM25(30%) + 重要性(10%)

L4: Files 持久化层（记忆归档）
    └── MEMORY.md + 每日日记 memory/

遗忘引擎：评分 = 访问频率×0.3 + 重要性×0.3 + 时效性×0.4
```

## 记忆生命周期

```
turn_start  → prefetch() 预取相关记忆
生成回复
turn_end    → sync_turn() 提取事实 + 写入记忆
session_end → on_session_end() 关键决策总结
心跳触发   → 遗忘检查 + 归档低分记忆
```

## 标准接口

```python
from src.memory_manager import MemoryHermes

mh = MemoryHermes()

# 存储记忆（自动 LLM 提取）
await mh.store("用户说石榴籽项目省赛过了", metadata={"project": "石榴籽"})

# 查询记忆
results = await mh.query("石榴籽项目进展")

# 预取（后台）
await mh.prefetch("优优的项目状态")

# 遗忘
await mh.forget(memory_id="mem_xxx")
```

## 配置

```json
{
  "embedding": {
    "provider": "dashscope",
    "model": "text-embedding-v3",
    "dimensions": 1024
  },
  "llm": {
    "provider": "bailian",
    "model": "qwen3.6-plus"
  },
  "decay": {
    "enabled": true,
    "threshold": 0.3,
    "check_interval_hours": 24
  },
  "layers": {
    "l1_compress": true,
    "l2_graph": true,
    "l3_vector": true,
    "l4_files": true
  }
}
```

---

_Memory Hermes · 融合 Hermes + Mem0 顶尖记忆技能_
