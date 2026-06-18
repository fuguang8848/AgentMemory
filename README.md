# AgentMemory

四层闭环记忆系统 (L1 工作记忆 / L2 短期 / L3 长期向量 / L4 知识图谱), 多 provider 自适应.

## Install (editable)

```bash
pip install -e .
```

## 快速使用

```python
from agentmemory import MemoryStore, MiniLMEmbedder
store = MemoryStore(embedder=MiniLMEmbedder())
store.add("今天学了 AgentMemory 的四层架构", user_id="浮光")
results = store.query("AgentMemory 架构", user_id="浮光", top_k=3)
```

## 文档

详见 `agentmemory/` 包内每个模块的 docstring.
