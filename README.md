# agentmemory

> 交响乐技能家族成员 | 顶尖四层闭环记忆系统

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)

融合 **Hermes** 记忆机制 + **Mem0** 混合检索优点，四层闭环记忆架构，为 AI Agent 提供持久化认知能力。

## 核心能力

| 能力 | 说明 |
|------|------|
| `memory.store` | 存储记忆（自动 LLM 事实提取） |
| `memory.query` | 查询记忆（混合检索：向量+BM25+重要性） |
| `memory.prefetch` | 预取相关记忆（后台异步加载） |
| `memory.forget` | 主动遗忘（评分驱动） |
| `memory.sync_turn` | 对话轮次同步（提取事实） |
| `memory.session_end` | 会话结束总结 |
| `memory.decay_check` | 遗忘引擎驱动检查 |
| `memory.stats` | 记忆系统统计 |

## 架构图

```
用户消息
    ↓
┌─────────────────────────┐
│  L1: LCM 压缩层          │ ← LLM 事实提取，不存原始对话
│  (对话 → 关键事实)        │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│  L2: Graph 图谱层         │ ← 实体关系（人名/项目/概念）
│  (事实 → 实体+关系)       │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│  L3: Vector 向量层        │ ← Mem0 混合检索
│  (事实 → 向量+BM25)       │   向量60% + BM25 30% + 重要性10%
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│  L4: Files 持久化层       │ ← MEMORY.md 长期记忆
│  (记忆 → 每日日记)        │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│  遗忘引擎                │ ← Mem0 风格遗忘算法
│  (低分记忆 → 归档/删除)   │   半衰期14天，阈值0.3
└─────────────────────────┘
```

## 安装

```bash
cd skills/agentmemory
pip install -e .
```

**依赖**：Python 3.10+，httpx

## 快速开始

```python
import asyncio
from src.memory_manager import MemoryHermes

mh = MemoryHermes()

async def main():
    # 存储记忆（自动 LLM 事实提取）
    await mh.store(
        "优优说石榴籽项目省赛结果要等几天",
        metadata={"source": "conversation"},
        importance=0.8
    )

    # 查询记忆
    results = await mh.query("石榴籽项目进展")
    for r in results:
        print(f"[{r['score']:.2f}] {r['content']}")

    # 预取（后台）
    prefetched = await mh.prefetch("优优的项目")

    # 遗忘检查
    decay_result = await mh.run_decay_check()
    print(f"遗忘: {decay_result['forget']}, 归档: {decay_result['archive']}")

asyncio.run(main())
```

## 记忆生命周期

```python
# 1. 对话开始前 - 预取相关记忆
prefetched = await mh.prefetch("优优")

# 2. 对话结束后 - 同步到记忆
stored = await mh.sync_turn(
    "省赛结果出了吗？",
    "还在审核中，预计下周出"
)
print(f"提取了 {len(stored)} 条事实")

# 3. 会话结束时 - 生成总结
stats = await mh.on_session_end()
print(f"会话持续 {stats['session_duration_seconds']}s")

# 4. 心跳触发 - 遗忘检查
decay_result = await mh.run_decay_check()
print(f"遗忘: {decay_result['forget']}, 归档: {decay_result['archive']}")
```

## 四层详解

### L1: LCM 压缩层
- LLM 提取关键事实（百炼 API）
- 不存原始对话内容
- 实体识别（人名/项目/日期/决策/偏好）
- 事实去重（相似度阈值 0.88）

### L2: Graph 图谱层
- 实体关系存储（人名/项目/概念/地点/组织）
- 支持 BFS 最短路径查询
- 邻居实体遍历
- 实体合并（别名支持）

### L3: Vector 向量层
- 混合检索：向量语义(60%) + BM25(30%) + 重要性(10%)
- 纯 Python BM25 实现（无外部依赖）
- dashscope text-embedding-v3 向量化

### L4: Files 持久化层
- MEMORY.md 长期记忆
- 每日日记 `memory/YYYY-MM-DD.md`
- 会话总结归档

## 遗忘算法

```
遗忘得分 = 访问频率(30%) + 重要性(30%) + 时效性(40%)

遗忘阈值 < 0.3  → 永久删除
归档阈值 0.3-0.5 → 归档到深层存储
保留阈值 > 0.5  → 保留
```

时效性衰减：`2^(-recency_days / half_life)`
- 14天后衰减到 50%

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
    "half_life_days": 14.0
  },
  "hybrid_search": {
    "vector_weight": 0.6,
    "bm25_weight": 0.3,
    "importance_weight": 0.1
  }
}
```

**环境变量**：
- `DASHSCOPE_API_KEY` - 向量化 API Key
- `BAILIAN_API_KEY` - LLM API Key

## 文件结构

```
agentmemory/
├── SKILL.md                  # OpenClaw 技能入口
├── README.md                 # 本文件
├── src/
│   ├── __init__.py
│   ├── config.py             # 配置管理
│   ├── memory_manager.py     # 总管理器
│   ├── L1_lcm_compressor.py  # L1 压缩层
│   ├── L2_graph_store.py     # L2 图谱层
│   ├── L3_vector_store.py    # L3 向量层
│   ├── L4_file_persist.py    # L4 文件层
│   ├── decay_engine.py        # 遗忘引擎
│   └── data/                 # 数据存储
│       ├── vectors.json
│       ├── graph_store.json
│       └── archive/
└── memory/                   # 每日日记
    └── YYYY-MM-DD.md
```

## 与市面方案对比

| 维度 | Mem0 | Letta | **agentmemory** |
|------|------|-------|----------------|
| 事实提取 | ✅ | ❌ | ✅ |
| 图谱层 | 混合 | PostgreSQL | ✅ Graph |
| 遗忘算法 | 重要性 | 压缩 | ✅ 完整 |
| 四层架构 | 混合 | 层级 | ✅ 闭环 |
| Prefetch | ❌ | ❌ | ✅ |
| 文件持久化 | ❌ | ❌ | ✅ |
| BM25 实现 | 外部 | 外部 | ✅ 纯 Python |

## 作为 OpenClaw 技能使用

```bash
openclaw skills install agentmemory
openclaw skills run agentmemory --action store --content "用户喜欢简洁回复"
openclaw skills run agentmemory --action query --query "用户偏好"
```

---

_MIT License | 交响乐技能家族成员_
