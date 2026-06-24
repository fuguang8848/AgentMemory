"""
MiniMax API Client — 清醒梦生成专用
=====================================

接入 MiniMax M3 (baseUrl=https://api.minimaxi.com/v1)
使用 Bearer token 认证，OpenAI 兼容接口。

使用示例：
    from minimax_client import MiniMaxClient

    client = MiniMaxClient(api_key="your-token", model="MiniMax-Text-01")
    response = client.chat("你好，生成一个创意想法")
"""

import os
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class MiniMaxError(Exception):
    """MiniMax API 错误"""
    pass


class MiniMaxClient:
    """
    MiniMax M3 API 客户端（OpenAI 兼容）

    Args:
        api_key: MiniMax API Bearer token
        model: 模型名，默认 MiniMax-Text-01
        base_url: API 地址，默认 https://api.minimaxi.com/v1
        timeout: 请求超时（秒），默认 60
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "MiniMax-Text-01",
        base_url: str = "https://api.minimaxi.com/v1",
        timeout: int = 60,
    ):
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        if not self.api_key:
            logger.warning("[MiniMax] API key 未设置，使用演示模式")

    # ─────────────────────────────────────────
    # 核心方法
    # ─────────────────────────────────────────

    def chat(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """
        发送对话请求到 MiniMax。

        Args:
            prompt: 用户输入
            system_prompt: 系统提示词
            temperature: 创造性温度
            max_tokens: 最大生成长度

        Returns:
            str: 模型生成的文本
        """
        if not self.api_key:
            return self._mock_response(prompt)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()

                return data["choices"][0]["message"]["content"]

        except httpx.HTTPStatusError as e:
            logger.error(f"[MiniMax] HTTP错误: {e.response.status_code} - {e.response.text}")
            raise MiniMaxError(f"API请求失败: {e.response.status_code}") from e
        except (KeyError, IndexError) as e:
            logger.error(f"[MiniMax] 响应解析错误: {e}")
            raise MiniMaxError(f"响应格式错误: {e}") from e
        except Exception as e:
            logger.error(f"[MiniMax] 请求异常: {e}")
            raise MiniMaxError(f"请求异常: {e}") from e

    # ─────────────────────────────────────────
    # 清醒梦专用生成
    # ─────────────────────────────────────────

    def generate_lucid_dream(self, prompt: str) -> str:
        """
        生成清醒梦内容（专用接口）

        使用较高的 temperature 以增加创意性。
        """
        system = "你是一个创意合成专家，擅长发现概念之间的意外联系。"
        return self.chat(
            prompt=prompt,
            system_prompt=system,
            temperature=0.85,
            max_tokens=1024,
        )

    # ─────────────────────────────────────────
    # 演示模式
    # ─────────────────────────────────────────

    @staticmethod
    def _mock_response(prompt: str) -> str:
        """无 API key 时的演示响应"""
        mock_responses = {
            "concept_fusion": """[关联分析] DreamNet 的知识图谱 + AgentTeam 的多Agent协作机制，可以构建一个"协作式梦境"——多个Agent在睡眠阶段共享记忆碎片，集体生成洞察。

[项目想法] 名称：CoDream（协作梦境系统）
- 目标：多Agent在夜间共享记忆图谱，协作生成项目洞察
- 方法：每个Agent的DreamNet输出作为边节点，通过AgentTeam的A2A协议共享
- 预期产出：每日早晨的"团队灵感简报"

[创新点] 首次将"协作学习"引入AI Agent的睡眠阶段""",

            "counterfactual": """[关键分支点] 当时如果选择了MiniMax M3而非Ollama，当前架构会有何不同？

[推理] M3的原生function calling和structured output能力相比Ollama更结构化，理论上可以减少工具调用的解析不确定性。

[建议] VCPToolBox可评估M3的原生工具协议，对比Ollama的工具兼容层，在工具调用密集场景做基准测试""",

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

        for key, resp in mock_responses.items():
            if key in prompt.lower():
                return resp

        return f"[灵感生成] 基于输入生成的创意想法: {prompt[:80]}..."


def create_minimax_client(
    api_key: Optional[str] = None,
    model: str = "MiniMax-Text-01",
) -> MiniMaxClient:
    """
    工厂函数：创建 MiniMax 客户端。

    优先使用传入的 api_key，其次环境变量 MINIMAX_API_KEY。
    """
    return MiniMaxClient(
        api_key=api_key,
        model=model,
    )
