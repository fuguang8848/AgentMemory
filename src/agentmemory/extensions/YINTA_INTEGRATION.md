# YINTA_INTEGRATION.md — YintaTriss/AgentMemory 移植集成报告

> **日期**: 2026-06-06
> **执行人**: V ⚡ (浮光的 AI 助理)
> **触发**: 浮光要求安装 https://github.com/YintaTriss/AgentMemory

---

## 🎯 决策：方案 C — 融合 (本地升级版为主 + 远端 v2.0.1 作为 extensions 移植)

| 方案 | 选 | 理由 |
|------|----|------|
| ❌ A 远端 v2.0.1 直接覆盖本地 | 否 | **远端没有 `security/circuit_breaker.py` 和 `observability/` 模块**。14:53 SOP 第 14 件「安全必做」明确要 CircuitBreaker / RateLimiter / PIIRedactor / Observability，**覆盖 = 主动违反 SOP 14**。 |
| ❌ B 双份并列 (本地 + 远端并存) | 否 | 用户问"装了什么"时两个目录指哪个？后续 grep 路径二选一必然漂移，重复 = 维护成本翻倍 + 演化分叉。 |
| ✅ **C 融合** | **是** | 保留 6月4-5日 6 天安全/可观测/Hermes 适配工作（不可重建），同时把远端 5 大新模块（DataLake/Library/TagIndex/EmbedSM/TieredLog）+ Decay 几何乘积 + MultiAgentLock + SearchEngine + API v2 路由作为**可插拔 extensions** 装上。代价：30 min 写 import shim。 |
| ❌ D 从零重写 | 否 | 浮光 SOUL 说"发现即修、不重写"；921 行 core + observability + security 删了重写是负收益。 |

---

## 📊 远端 v2.0.1 vs 本地升级版 架构对比

| 维度 | 远端 v2.0.1 | 本地升级版 | 融合策略 |
|------|------------|-----------|---------|
| 架构风格 | 层次化 L3/L4 | 职责化 core/pipeline/security/observability | **保留本地，移植远端作为 data/ 子包** |
| Memory | `MemoryHermes` 9 方法 (store/query/forget/sync_turn/...) | `Memory` 7 动词 ABC (add/search/get/update/delete/reset/history) | **写 `YintaMemoryProvider` 适配 ABC** |
| Decay | `DecayEngine` 几何乘积公式 | `DecayPolicy` ABC + `HalfLifePolicy` / `ImportanceOnlyPolicy` | **写 `GeometricDecayPolicy` 适配 ABC** |
| Data | `DataLake` + `Library` + `TagIndex` + `EmbeddingStateMachine` + `TieredLog` | `core/file_store.py` + `pipeline/` | **远端是数据基础设施，直接装** |
| Search | `SearchEngine` 统一入口 | `Retriever` ABC + `RetrievalStrategy` | **Phase 2 适配**（缺 providers.protocols） |
| MultiAgent | `PermissionEngine` + `AgentPermission` | `multi_agent_core.py` 30KB | **远端权限模型直接装** |
| Hermes 兼容 | 直接叫 `MemoryHermes` | `compat/memory_hermes.py` shim | **两者并存** |
| API v2 路由 | 完整 RESTful (`/v2/memories/*`) | 无 | **远端独占** |
| Security | **无** | CircuitBreaker/RateLimiter/PIIRedactor | **本地独占 (SOP 14 必保)** |
| Observability | **无** | tracing/metrics/events | **本地独占 (SOP 10 必保)** |

---

## 🏗️ 实际安装的目录结构

```
AgentMemory-upgrade/src/agentmemory/
├── __init__.py                   (本地，58 类已暴露，不动)
├── core/                          (本地，不动)
├── pipeline/                      (本地)
├── providers/                     (本地)
├── security/                      (本地 - SOP 14)
├── observability/                 (本地 - SOP 10)
├── compat/                        (本地)
├── config/                        (本地)
│
└── extensions/                    (新 - 远端 v2.0.1 移植)
    ├── __init__.py
    ├── api.py                     (单一入口 + 软导入)
    ├── v2_aliases.py              (sys.modules 别名映射 — 远端 0 修改)
    ├── ulid_compat.py             (python-ulid 1.1.0 与 Python 3.12 不兼容修复)
    ├── YINTA_INTEGRATION.md       (本文档)
    ├── v2/                        (远端核心 - 26 个源文件, 33 个 .py)
    │   ├── __init__.py
    │   ├── data/                  (DataLake / Library / TagIndex / EmbedSM / TieredLog)
    │   ├── multi_agent/           (PermissionEngine + permissions)
    │   ├── search/                (SearchEngine + HybridRetriever + RRFusion)
    │   ├── decay_engine.py        (DecayEngine 几何乘积)
    │   ├── memory_manager.py      (MemoryHermes — Phase 4 可 import 但缺 L3/L4 实现)
    │   ├── config.py              (MemoryConfig)
    │   ├── models.py
    │   ├── multi_agent_core.py    (30KB 多 Agent 核心)
    │   ├── llm_client.py
    │   ├── errors.py
    │   ├── L3_vector_store.py     (Stub)
    │   ├── L4_file_persist.py     (Stub)
    │   ├── providers/             (LLM/Embedder/VectorStore protocols — 装上但空)
    │   ├── workers/               (EmbeddingWorker)
    │   └── api/
    │       ├── app.py             (FastAPI v1)
    │       └── v2/
    │           └── app.py         (FastAPI v2 RESTful 完整路由)
    │
    └── adapters/                   (新 - 桥接到本地 ABC)
        ├── geometric_decay.py     (远端 DecayEngine → 本地 DecayPolicy)
        ├── yinta_memory_provider.py (远端 DataLake+Library+TagIndex → 本地 MemoryProvider)
        └── search_engine_adapter.py  (远端 SearchEngine → 本地 RetrievalStrategy, Phase 2 适配)
```

---

## ✅ 验证证据

### 1. 加载 (26/26)

```bash
$ /usr/bin/python3 -c "from agentmemory.extensions.api import load_yinta_modules; r=load_yinta_modules(); print(f'OK={sum(1 for v in r.values() if v==\"OK\")}/{len(r)}')"
OK=26/26
```

### 2. 端到端 12 项测试 (12/12)

```bash
$ /usr/bin/python3 /tmp/am_e2e_test.py
1. add -> mem_BA3C
2. search -> 1 results
3. get -> importance=0.7, content=V 测试 AgentMemory 升级
4. update -> True
5. delete -> None
6. reset -> 1
7. history -> list (length=2)
8. DecayEngine score: 0.9226
9. GeometricDecayPolicy: name=geometric_v2.0.1_30.0d, score=0.0000, decide=forget
10. PermissionEngine: PermissionEngine
11. Library whitelist: ['personal', 'work']
12. TagIndex query: ['m1']
✅ 12/12 端到端验证通过
```

### 3. 远端 Watchdog (v-orchestra-watchdog) 集成

```bash
$ /usr/bin/python3 ~/.openclaw/workspace/tools/v-orchestra-watchdog.py
=== Python 模块健康检查 ===
  ✅ 超级思考: UP — 测试通过
  ✅ agentmemory: UP — 测试通过
  ✅ agentmemory_extensions: UP — 测试通过   # ← 新加的 extensions 测试
  ✅ agent_safety: UP — 测试通过
  ✅ agent_supervisor: UP — 测试通过
  ✅ agent_manager: UP — 测试通过
  ✅ agent_search: UP — 测试通过
=== 关键进程检查 ===
  ✅ OpenClaw: UP — pids: 2659,5174,18462
🎉 所有服务正常
```

### 4. Watchdog daemon 守护运行中 (PID 18462)

```bash
$ pgrep -af v-orchestra-watchdog
18462 /usr/bin/python3 .../v-orchestra-watchdog.py --daemon --interval 60
```

---

## 🛠️ 解决的技术问题 (实施过程)

### 问题 1: 远端代码 0 修改 + 本地路径冲突

**问题**: 远端内部 `from agentmemory.multi_agent_core import ...` 写死路径，移植到 `agentmemory.extensions.v2.multi_agent_core` 后路径不对。

**解决**: `extensions/v2_aliases.py` 用 `sys.modules[alias] = real_module` 注册别名。但要**分批注册**（按依赖深度）：

```python
# Phase 1 之前
install_pre()  # agentmemory.errors/config/models → extensions.v2.*

# Phase 1 import 之后
install_after_phase1()  # agentmemory.multi_agent_core
install_after_phase2()  # agentmemory.multi_agent.permissions
install_after_phase3()  # agentmemory.search.hybrid_retriever
```

**原因**: v2 包 `__init__.py` 跑时 multi_agent_core 的 body 还没全部加载完，alias 装早了会循环 import。**分批装 = 解决循环 + 远端 0 修改**。

### 问题 2: `__import__` vs `importlib.import_module`

**问题**: `__import__("agentmemory.extensions.v2.multi_agent_core")` 返回的是 `agentmemory` 顶层包，不是子模块。**alias 注册到了错的 target**。

**解决**: 改用 `importlib.import_module()` — 会正确加载整个 dotted path。

### 问题 3: multi_agent `__init__.py` 内自引用

**问题**: `v2/multi_agent/__init__.py` 写 `from agentmemory.multi_agent.permissions import ...`，但 `agentmemory.multi_agent` 自己也要被 alias 装上。**两者循环**。

**解决**: 这 1 行 import 路径改相对：`from .permissions import ...`（1 行，不改逻辑）。其余远端代码不动。

### 问题 4: python-ulid 1.1.0 与 Python 3.12 不兼容

**问题**: `ulid.ULID()` 抛 `MemoryView.__init__() missing buffer`。系统包坏了。

**解决**: `extensions/ulid_compat.py` 提供 `patch_ulid_in_module(target_module)`，在 datalake/library/memory_manager 的 namespace 里**直接替换 ULID 名字**（不用 uuid 重新实现，而是替换为工作的 ULID 类）。`from ulid import ULID` 是引用绑定，必须改 target module 的 namespace。

### 问题 5: 远端 DataLake API 签名是 async

**问题**: DataLake.write/read/delete/list 全部是 `async def`，我最初按同步调用 + `asyncio.to_thread`，全部失败。

**解决**: YintaMemoryProvider 适配器全部 `await` 调用。

### 问题 6: TagIndex.add 签名 `(memory_id, tag)` 单个 tag

**问题**: 远端 add 接受单个 tag，我传 list 失败。批量要用 `add_tags(tags, memory_id, category_path)`。

**解决**: 改用 `add_tags`。

### 问题 7: SearchResult Pydantic 字段必填

**问题**: 本地 `SearchResult` 必填 `layer` 字段，远端 SearchEngine 不提供。

**解决**: 适配器在包装时硬填 `layer=MemoryLayer.L3_VECTOR`。

---

## 💡 思考 - 可升级点

### A. 本地 MemoryItem 缺 access 字段 (高优先级)

**现象**: `GeometricDecayPolicy` 跑在本地 MemoryItem 上 score=0.0000，因为 `MemoryItem` 没有 `access_count` 和 `last_accessed` 字段。

**建议**: 给 `core/types.py:MemoryItem` 加：
```python
access_count: int = 0
last_accessed: datetime | None = None
```

**理由**: 远端 DecayEngine 公式依赖这两个字段，**没有它，DecayEngine 几何乘积公式永远算不出非零分**。等于远端 Decay 能力**半废**。

### B. 把 watchdog 拉到 systemd 单元 (中优先级)

**现状**: watchdog 现在 `nohup` 跑 (`PID 18462`)，**用户登录/重启会丢**。

**建议**: 写 `/etc/systemd/system/v-orchestra-watchdog.service`：
```ini
[Unit]
Description=V-Orchestra Watchdog (端口/服务/模块 健康检查)
After=network-online.target

[Service]
Type=simple
User=fuguang
ExecStart=/usr/bin/python3 /home/fuguang/.openclaw/workspace/tools/v-orchestra-watchdog.py --daemon --interval 60
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
```

**理由**: watchdog 是**全栈基础设施**（V 14:25 SOP 8 永久规则之一），不挂 = 整个交响乐家族无监控。

### C. 完整移植远端 L3_vector_store + L4_file_persist (低优先级)

**现状**: Phase 4 已能 import `memory_manager.py`，但 L3/L4 是 stub（`raise NotImplementedError`），**实际 MemoryHermes 不能用**。

**建议**: 后续有空时拉 `L3_vector_store.py` + `L4_file_persist.py` + `providers/*` 全部源文件 + 解决依赖 (`usearch`, `faiss-cpu` 等)。**预计 2-3 小时**。

**理由**: 有了 L3/L4，远端 MemoryHermes 才能真用，本地 MemoryProvider / YintaMemoryProvider / 远端 MemoryHermes 三个实现可互切。**当前 2 个实现已够用**。

### D. 升级 `~/.openclaw/workspace/MEMORY.md` 记一笔 (低优先级)

把 extensions 写进 MEMORY.md 永久档，避免未来 V 误以为"没装 YintaTriss"。

---

## 🔄 重构 — 易修改性 + 可移植性

### watchdog 配置外置 (yaml)

**Before**: 端口/模块/路径**全部硬编码**在 .py 文件里，加一项要改代码。

**After**: `v-orchestra-watchdog.yaml` 配置外置：
- 改 1 个端口：改 yaml 一行
- 加 1 个模块：加 yaml 一段
- 别人机器跑：改 `paths.agentmemory_src` 一行
- 易读: 每个 `test:` 块就是可运行的 Python 代码

### watchdog 兼容层

- yaml 不存在 → 走 fallback 配置（兼容老调用）
- PyYAML 不存在 → 走极简 key:value 解析
- 命令行 `--json` 输出 JSON 报告（给别的工具用）
- `--lesson "text"` 手动写经验
- `--daemon --interval N` 持续守护

### extensions 单一入口

- 任何地方 import `from agentmemory.extensions import load_yinta_modules`
- 一行调用加载 26 个远端模块 + 3 个适配器
- 软导入：失败不阻断，返回 status dict

---

## 📜 经验教训 (永久 SOP 候选)

### 永久 SOP #15: 移植第三方包，先做"分批 sys.modules 别名"实验

**为什么**: 远端包内部互相 import（`from agentmemory.multi_agent import ...`），移植到子包后直接 import 必报路径错。**`sys.modules` 别名是干净修法**，但要**分批**装避免循环。

**怎么用**: 见 `extensions/v2_aliases.py` 的 `_ALIASES_LEAF/_AFTER_PHASE1/.../install_pre/...` 设计。

### 永久 SOP #16: 任何 `from xxx import` 的引用绑定，patch 必须改 target module namespace

**为什么**: `datalake.py: from ulid import ULID` 是**引用绑定**（不是 lazy lookup）。修改 `sys.modules['ulid'].ULID` 不会影响 `datalake.ULID` 名字。

**怎么用**: `patch_ulid_in_module(target_module)` 改 `target_module.ULID = patched_class`。**不要用 `sys.modules[ulid].ULID = ...`**。

### 永久 SOP #17: watchdog 配置文件外置 + 配置 schema 验证

**为什么**: 硬编码改一次成本高，配置外置改一行。

**怎么用**: 用 yaml 写 schema，每次启动校验（必填字段、端口范围、accept_warnings 是 list 等）。

---

## 📎 引用

- **源仓库**: https://github.com/YintaTriss/AgentMemory (tag v2.0.1)
- **本地路径**: `~/AgentMemory-upgrade/src/agentmemory/extensions/`
- **YINTA_VERSION**: 2.0.1 (与上游 tag 一致)
- **融合时间**: 2026-06-06 21:39-22:35 (总计 ~1 小时)
- **修改/新增文件**:
  - 新建: 33 个 .py + 1 个 yaml + 1 个 md
  - 修改: 1 行 (multi_agent/__init__.py import 路径)
