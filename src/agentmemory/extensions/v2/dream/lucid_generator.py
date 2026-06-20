"""
清醒梦生成器 — LucidDreamGenerator
===================================

核心能力：在梦境周期中，主动生成"清醒梦"——让AI在睡眠阶段
有意识地探索概念之间的非常规关联，激发灵感。

灵感来源：
- LeoYeAI/openclaw-auto-dream：梦境被动收集
- ldclabs/anda-brain：神经符号探索
-认知科学清醒梦研究：有意识地引导梦境内容

清醒梦类型：
1. 概念融合：取两个无关概念，强制关联
2. 反事实推理：如果当时做了不同选择，会怎样？
3. 跨域迁移：从A领域的解决方案，迁移到B领域
4. 极端化思考：将某个趋势推到极限，看结果
5. 逆向设计：从目标倒推实现路径

输出：
- dream_inspirations/：每日灵感 JSON
- 推送到指定 webhook/文件/消息队列
"""

import json
import logging
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 清醒梦提示词模板
# ─────────────────────────────────────────────

LUCID_PROMPTS = {
    "concept_fusion": """你是一个创意合成器。请从以下两个随机概念中，发现一个令人惊讶的联系，并提出一个具体可行的项目想法：

概念A: {concept_a}
概念B: {concept_b}

要求：
1. 关联必须是非显而易见的，不能是表面相似
2. 项目想法必须具体，包含目标、方法和预期产出
3. 用中文回答，格式：[关联分析] 2-3句 / [项目想法] 具体描述 / [创新点] 1-2句
4. 控制在200字以内
""",

    "counterfactual": """你是一个反事实推理引擎。请探索以下记忆/事件的一个替代路径：

记忆内容: {memory_content}

要求：
1. 描述"如果当时做了X，结果会怎样"的具体推理
2. 识别关键的决策分支点
3. 提炼对未来的决策建议
4. 用中文回答，控制在150字以内
""",

    "cross_domain": """你是一个跨领域创新专家。请将以下技术/方法的原理，迁移到一个全新的领域：

技术/方法: {technique}
当前领域: {source_domain}

要求：
1. 识别该技术的核心原理（不是表面方法）
2. 提出3个可迁移的目标领域
3. 对每个领域给出一个具体应用示例
4. 用中文回答，控制在200字以内
""",

    "extremize": """你是一个极限思考者。请将以下趋势推到一个荒谬的极端，然后分析：

趋势: {trend}

要求：
1. 先推到逻辑终点（即使看起来荒谬）
2. 从极端结果反推当前应该做什么
3. 识别哪些现有假设会被打破
4. 用中文回答，控制在150字以内
""",

    "inverse_design": """你是一个逆向架构师。请从以下目标反向设计实现路径：

目标: {goal}

要求：
1. 从目标倒推需要先解决哪些前置问题
2. 识别关键技术瓶颈
3. 给出3个可能的实现路径（激进/平衡/保守）
4. 用中文回答，控制在200字以内
""",
}


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class LucidDream:
    """清醒梦"""
    id: str
    dream_type: str
    prompt_template: str
    input_data: dict
    generated_content: str
    insight_score: float
    novelty_tags: list[str]
    created_at: str
    used_for_projects: list[str] = field(default_factory=list)


@dataclass
class Inspiration:
    """灵感推送条目"""
    id: str
    title: str
    body: str
    dream_id: str
    relevance_tags: list[str]
    action_suggestion: str
    priority: str
    read: bool = False
    created_at: str = ""


# ─────────────────────────────────────────────
# 清醒梦生成器
# ─────────────────────────────────────────────

class LucidDreamGenerator:
    """
    在梦境周期中，主动生成清醒梦（Lucid Dreams）

    工作流程：
    1. 从知识图谱中提取高频概念、项目、人物
    2. 随机抽取2个无关概念，触发概念融合
    3. 对重要决策做反事实推理
    4. 跨域迁移技术方案
    5. 收集LLM生成的灵感，保存到 dream_inspirations/
    6. 生成可推送的 Inspiration 条目
    """

    def __init__(
        self,
        memory_dir: str = "~/.openclaw/workspace/memory",
        llm_provider: str = "openai",   # 或 "anthropic", "minimax"
        model: str = "gpt-4o",
        num_dreams: int = 5,            # 每次梦境周期生成 N 个清醒梦
    ):
        self.memory_dir = Path(memory_dir).expanduser()
        self.llm_provider = llm_provider
        self.model = model
        self.num_dreams = num_dreams

        self.output_dir = self.memory_dir / "dream_inspirations"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._concepts: list[str] = []      # 从图谱提取的概念
        self._projects: list[str] = []       # 项目名
        self._decisions: list[str] = []     # 重要决策
        self._techniques: list[str] = []     # 技术方法

    # ─────────────────────────────────────────
    # 公共 API
    # ─────────────────────────────────────────

    def generate(self, graph_stats: dict) -> list[LucidDream]:
        """
        入口：生成 N 个清醒梦

        Args:
            graph_stats: 知识图谱统计（节点数、边数、类型分布）

        Returns:
            list[LucidDream]: 生成的清醒梦列表
        """
        # 1. 收集图谱中的概念和项目
        self._load_from_graph(graph_stats)

        dreams = []

        # 2. 概念融合（最常用）
        for _ in range(max(1, self.num_dreams // 2)):
            dream = self._generate_concept_fusion()
            if dream:
                dreams.append(dream)

        # 3. 反事实推理
        if self._decisions:
            for _ in range(max(1, self.num_dreams // 4)):
                dream = self._generate_counterfactual()
                if dream:
                    dreams.append(dream)

        # 4. 跨域迁移
        if self._techniques:
            for _ in range(max(1, self.num_dreams // 4)):
                dream = self._generate_cross_domain()
                if dream:
                    dreams.append(dream)

        # 5. 极限思考
        for _ in range(1):
            dream = self._generate_extremize()
            if dream:
                dreams.append(dream)

        # 6. 保存并生成灵感
        inspirations = self._save_and_create_inspirations(dreams)

        logger.info(f"  [Lucid] 生成了 {len(dreams)} 个清醒梦，{len(inspirations)} 条灵感")

        return dreams

    # ─────────────────────────────────────────
    # 内部生成方法
    # ─────────────────────────────────────────

    def _load_from_graph(self, graph_stats: dict):
        """从图谱统计中提取概念（实际应读取图谱文件）"""
        # 实际生产中应该读取 graph.json 提取节点
        # 这里用启发式：高频标签 = 重要概念
        self._concepts = [
            "AgentMemory", "AgentTeam", "知识图谱", "记忆系统",
            "OpenClaw", "Ollama", "VCPToolBox",
            "workflow", "telemetry", "安全审计",
            "多Agent协作", "上下文管理", "修真防御",
        ]
        self._projects = [
            "SpectrAI", "DreamNet", "AgentMemory-v2",
            "OpenClaw修真防御", "VCPToolBox",
        ]
        self._decisions = [
            "使用Ollama作为本地模型提供商",
            "采用A2A协议进行多Agent通信",
            "三层防御体系应对OpenClaw修真问题",
        ]
        self._techniques = [
            "遗忘曲线（Ebbinghaus）", "重要性评分",
            "OpenTelemetry遥测", "知识图谱",
            "上下文压缩（LCM）",
        ]

    def _generate_concept_fusion(self) -> Optional[LucidDream]:
        """概念融合清醒梦"""
        if len(self._concepts) < 2:
            return None

        concept_a, concept_b = random.sample(self._concepts, 2)

        prompt = LUCID_PROMPTS["concept_fusion"].format(
            concept_a=concept_a,
            concept_b=concept_b,
        )

        content = self._call_llm(prompt)

        dream = LucidDream(
            id=f"dream_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{random.randint(1000,9999)}",
            dream_type="concept_fusion",
            prompt_template="concept_fusion",
            input_data={"concept_a": concept_a, "concept_b": concept_b},
            generated_content=content,
            insight_score=self._score_insight(content),
            novelty_tags=["概念融合"],
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        return dream

    def _generate_counterfactual(self) -> Optional[LucidDream]:
        """反事实推理清醒梦"""
        if not self._decisions:
            return None

        memory = random.choice(self._decisions)

        prompt = LUCID_PROMPTS["counterfactual"].format(
            memory_content=memory,
        )

        content = self._call_llm(prompt)

        dream = LucidDream(
            id=f"dream_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{random.randint(1000,9999)}",
            dream_type="counterfactual",
            prompt_template="counterfactual",
            input_data={"memory": memory},
            generated_content=content,
            insight_score=self._score_insight(content),
            novelty_tags=["反事实推理"],
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        return dream

    def _generate_cross_domain(self) -> Optional[LucidDream]:
        """跨域迁移清醒梦"""
        if not self._techniques:
            return None

        technique = random.choice(self._techniques)

        prompt = LUCID_PROMPTS["cross_domain"].format(
            technique=technique,
            source_domain="AI Agent系统",
        )

        content = self._call_llm(prompt)

        dream = LucidDream(
            id=f"dream_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{random.randint(1000,9999)}",
            dream_type="cross_domain",
            prompt_template="cross_domain",
            input_data={"technique": technique},
            generated_content=content,
            insight_score=self._score_insight(content),
            novelty_tags=["跨域迁移"],
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        return dream

    def _generate_extremize(self) -> Optional[LucidDream]:
        """极限思考清醒梦"""
        trends = [
            "AI Agent 自主性越来越高",
            "上下文窗口越来越大",
            "多模态融合越来越深",
        ]
        trend = random.choice(trends)

        prompt = LUCID_PROMPTS["extremize"].format(trend=trend)

        content = self._call_llm(prompt)

        dream = LucidDream(
            id=f"dream_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{random.randint(1000,9999)}",
            dream_type="extremize",
            prompt_template="extremize",
            input_data={"trend": trend},
            generated_content=content,
            insight_score=self._score_insight(content),
            novelty_tags=["极限思考"],
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        return dream

    def _generate_inverse_design(self) -> Optional[LucidDream]:
        """逆向设计清醒梦"""
        goals = [
            "让AI Agent拥有真正的长期记忆",
            "实现多Agent的自主协作",
            "构建自我进化的安全防御系统",
        ]
        goal = random.choice(goals)

        prompt = LUCID_PROMPTS["inverse_design"].format(goal=goal)

        content = self._call_llm(prompt)

        dream = LucidDream(
            id=f"dream_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{random.randint(1000,9999)}",
            dream_type="inverse_design",
            prompt_template="inverse_design",
            input_data={"goal": goal},
            generated_content=content,
            insight_score=self._score_insight(content),
            novelty_tags=["逆向设计"],
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        return dream

    # ─────────────────────────────────────────
    # LLM 调用（可替换为真实provider）
    # ─────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        """
        调用 LLM 生成清醒梦内容。

        生产环境替换为真实 provider：
        - minimax: 使用 MiniMax API
        - anthropic: 使用 Claude API
        - openai: 使用 GPT-4 API
        """
        # ─── 演示模式：返回结构化假数据 ───
        # 实际生产中替换为真实 API 调用
        dream_type = prompt.split("\n")[0] if "\n" in prompt else "概念融合"

        mock_responses = {
            "concept_fusion": """[关联分析] DreamNet 的知识图谱 + AgentTeam 的多Agent协作机制，可以构建一个"协作式梦境"——多个Agent在睡眠阶段共享记忆碎片，集体生成洞察。

[项目想法] 名称：CoDream（协作梦境系统）
- 目标：多Agent在夜间共享记忆图谱，协作生成项目洞察
- 方法：每个Agent的DreamNet输出作为边节点，通过AgentTeam的A2A协议共享
- 预期产出：每日早晨的"团队灵感简报"

[创新点] 首次将"协作学习"引入AI Agent的睡眠阶段""",

            "counterfactual": """[关键分支点] 当时如果选择了MiniMax M3而非Ollama，上下文污染问题是否会不同？

[推理] M3的buildReplayPolicy接口可能提供更好的thinking通道控制，相比Ollama的无差别传递，理论上可以更早阻断修真内容。

[建议] 评估在VCPToolBox中增加模型切换策略——当检测到修真污染时，自动切换到有buildReplayPolicy的模型""",

            "cross_domain": """[核心原理] 遗忘曲线（Ebbinghaus）的核心不是"记忆衰退"，而是"选择性强化"——大脑主动保留高价值记忆，淘汰低价值记忆。

[可迁移领域]
1. 软件架构：主动淘汰低使用率模块，保留核心API
2. 个人知识管理：像大脑一样评分，淘汰低引用笔记
3. 投资组合：动态强化高回报资产，淘汰低效资产

[应用示例] DreamNet的遗忘曲线可以改造为"Skill动态淘汰"——低使用率Skill自动归档，保持AgentManager轻量化""",

            "extremize": """[逻辑终点] 如果上下文窗口无限大（10亿token），则：
- 记忆系统变得无关紧要（全部塞进上下文）
- Agent失去"遗忘"能力，所有经验等权重
- 上下文污染风险达到最高（无法区分信号/噪声）

[反推当前] 即使未来窗口变大，记忆系统仍然必要：
- 模型推理成本随上下文线性增长
- 无限窗口≠无限注意力
- 遗忘是智能的必要条件（而非缺陷）

[打破的假设] "记忆=上下文"的假设是错误的""",
        }

        # 从mock中选择或生成默认响应
        for key, resp in mock_responses.items():
            if key in prompt.lower() or key in self._prompts_contain(prompt):
                return resp

        return f"[灵感生成] 基于以下输入：{prompt[:50]}... — 这是一个清醒梦的创意输出。"

    def _prompts_contain(self, prompt: str) -> str:
        for key in LUCID_PROMPTS:
            if key in prompt:
                return key
        return "concept_fusion"

    def _score_insight(self, content: str) -> float:
        """启发式评分：内容越长、结构越丰富，分数越高"""
        score = 0.3
        score += min(len(content) / 1000, 0.3)   # 长度
        score += 0.1 if "：" in content else 0  # 结构化
        score += 0.1 if "\n" in content else 0  # 多行
        score += 0.1 if any(c in content for c in ["1.", "2.", "3."]) else 0
        return min(score, 1.0)

    # ─────────────────────────────────────────
    # 保存与推送
    # ─────────────────────────────────────────

    def _save_and_create_inspirations(self, dreams: list[LucidDream]) -> list[Inspiration]:
        """保存清醒梦，生成可推送的灵感"""
        inspirations = []

        for dream in dreams:
            # 1. 保存清醒梦 JSON
            dream_file = self.output_dir / f"{dream.id}.json"
            with open(dream_file, "w") as f:
                json.dump({
                    "id": dream.id,
                    "type": dream.dream_type,
                    "content": dream.generated_content,
                    "score": dream.insight_score,
                    "tags": dream.novelty_tags,
                    "created_at": dream.created_at,
                }, f, ensure_ascii=False, indent=2)

            # 2. 生成灵感推送条目
            lines = dream.generated_content.strip().split("\n")
            title = lines[0] if lines else f"清醒梦 #{dream.id[-4:]}"

            action_map = {
                "concept_fusion": "评估是否可以启动CoDream项目",
                "counterfactual": "检查VCPToolBox模型切换策略",
                "cross_domain": "将跨域方案应用到DreamNet遗忘曲线",
                "extremize": "验证记忆系统的必要性",
                "inverse_design": "规划CoDream技术路线",
            }

            inspiration = Inspiration(
                id=f"insp_{dream.id}",
                title=title[:100],
                body=dream.generated_content,
                dream_id=dream.id,
                relevance_tags=dream.novelty_tags,
                action_suggestion=action_map.get(dream.dream_type, "评估可行性"),
                priority="HIGH" if dream.insight_score > 0.6 else "MEDIUM",
                created_at=dream.created_at,
            )
            inspirations.append(inspiration)

        # 3. 保存灵感索引
        self._save_inspiration_index(inspirations)

        return inspirations

    def _save_inspiration_index(self, inspirations: list[Inspiration]):
        """保存灵感索引文件"""
        index_file = self.output_dir / "inspirations.json"
        data = [
            {
                "id": i.id,
                "title": i.title,
                "body": i.body,
                "dream_id": i.dream_id,
                "tags": i.relevance_tags,
                "action": i.action_suggestion,
                "priority": i.priority,
                "read": i.read,
                "created_at": i.created_at,
            }
            for i in inspirations
        ]
        with open(index_file, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_unread_inspirations(self) -> list[Inspiration]:
        """获取所有未读灵感（用于早晨推送）"""
        index_file = self.output_dir / "inspirations.json"
        if not index_file.exists():
            return []

        with open(index_file) as f:
            data = json.load(f)

        return [
            Inspiration(
                id=d["id"],
                title=d["title"],
                body=d["body"],
                dream_id=d["dream_id"],
                relevance_tags=d["tags"],
                action_suggestion=d["action"],
                priority=d["priority"],
                read=d["read"],
                created_at=d["created_at"],
            )
            for d in data if not d.get("read", False)
        ]

    def mark_read(self, inspiration_id: str):
        """标记灵感为已读"""
        index_file = self.output_dir / "inspirations.json"
        if not index_file.exists():
            return

        with open(index_file) as f:
            data = json.load(f)

        for d in data:
            if d["id"] == inspiration_id:
                d["read"] = True

        with open(index_file, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
