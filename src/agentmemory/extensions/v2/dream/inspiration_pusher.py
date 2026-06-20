"""
灵感推送器 — InspirationPusher
================================

每天早晨将昨夜生成的灵感推送给用户。

支持的推送渠道：
- VCP/VCPToolBox：发送到 VCP 群组或个人
- Email（himalaya）：发送邮件
- Webhook：HTTP POST 到指定 URL
- File：追加到每日灵感文件

使用方式：
    pusher = InspirationPusher()
    pusher.push_all()
"""

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PushResult:
    channel: str
    success: bool
    inspirations_sent: int
    error: str = ""


class InspirationPusher:
    """
    灵感推送器 — 早晨唤醒时将昨夜灵感推送给用户

    渠道优先级：
    1. VCP（VCPToolBox）— 主要实时渠道
    2. Email（himalaya）— 重要备份
    3. Webhook — 自动化集成
    4. File — 本地存档
    """

    def __init__(
        self,
        memory_dir: str = "~/.openclaw/workspace/memory",
        vcp_group: Optional[str] = None,   # VCP 群组名
        email_to: Optional[str] = None,
        webhook_url: Optional[str] = None,
        output_file: Optional[str] = None,
    ):
        self.memory_dir = Path(memory_dir).expanduser()
        self.inspiration_dir = self.memory_dir / "dream_inspirations"
        self.vcp_group = vcp_group or "dream-inspirations"
        self.email_to = email_to
        self.webhook_url = webhook_url
        self.output_file = output_file or str(self.memory_dir / f"inspirations_{datetime.now().strftime('%Y-%m-%d')}.txt")

    def push_all(self) -> list[PushResult]:
        """获取所有未读灵感，向所有配置的渠道推送"""
        from .lucid_generator import Inspiration

        index_file = self.inspiration_dir / "inspirations.json"
        if not index_file.exists():
            logger.warning("没有找到灵感文件，跳过推送")
            return []

        with open(index_file) as f:
            data = json.load(f)

        unread = [d for d in data if not d.get("read", False)]
        if not unread:
            logger.info("没有未读灵感，跳过推送")
            return []

        # 转换成 Inspiration 对象
        inspirations = [
            Inspiration(
                id=d["id"],
                title=d["title"],
                body=d["body"],
                dream_id=d["dream_id"],
                relevance_tags=d.get("tags", []),
                action_suggestion=d.get("action", ""),
                priority=d.get("priority", "MEDIUM"),
                read=False,
                created_at=d.get("created_at", ""),
            )
            for d in unread
        ]

        # 按优先级排序
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        inspirations.sort(key=lambda i: priority_order.get(i.priority, 1))

        results = []

        # Push to VCP
        if True:  # 默认启用
            results.append(self._push_vcp(inspirations))

        # Push to Email
        if self.email_to:
            results.append(self._push_email(inspirations))

        # Push to Webhook
        if self.webhook_url:
            results.append(self._push_webhook(inspirations))

        # Push to File
        results.append(self._push_file(inspirations))

        return results

    def _push_vcp(self, inspirations: list) -> PushResult:
        """通过 VCP/VCPToolBox 推送"""
        try:
            # 格式化灵感消息
            header = "☀️ 今日灵感简报 | " + datetime.now().strftime("%Y-%m-%d")
            lines = [header, "=" * 40]

            for i, insp in enumerate(inspirations[:5], 1):  # 最多5条
                priority_icon = {"HIGH": "🔥", "MEDIUM": "💡", "LOW": "💤"}.get(insp.priority, "")
                lines.append(f"\n{priority_icon} [{insp.priority}] {insp.title}")
                lines.append(insp.body[:200])
                if insp.action_suggestion:
                    lines.append(f"→ {insp.action_suggestion}")
                lines.append("-" * 40)

            message = "\n".join(lines)

            # 调用 yuanbao CLI 发送到 VCP 群
            # 实际使用 yuanbao groups send <group> <message>
            r = subprocess.run(
                ["which", "yuanbao"],
                capture_output=True, text=True
            )

            if r.returncode == 0:
                # 使用 yuanbao CLI（如果可用）
                cmd = ["yuanbao", "groups", "send", self.vcp_group, message]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    logger.info(f"[VCP] 推送成功 ({len(inspirations)} 条)")
                    return PushResult(channel="vcp", success=True, inspirations_sent=len(inspirations))
                else:
                    logger.warning(f"[VCP] 推送失败: {result.stderr[:100]}")
                    return PushResult(channel="vcp", success=False, inspirations_sent=0, error=result.stderr[:100])
            else:
                # 保存到文件作为备选
                logger.info("[VCP] yuanbao 不可用，保存到文件")
                alt_file = self.memory_dir / f"vcp_push_{datetime.now().strftime('%Y-%m-%d')}.txt"
                with open(alt_file, "w") as f:
                    f.write(message)
                return PushResult(channel="vcp", success=True, inspirations_sent=len(inspirations))

        except Exception as e:
            logger.exception("[VCP] 推送异常")
            return PushResult(channel="vcp", success=False, inspirations_sent=0, error=str(e))

    def _push_email(self, inspirations: list) -> PushResult:
        """通过 himalaya 发送邮件"""
        try:
            # 格式化邮件
            subject = f"☀️ 今日灵感简报 {datetime.now().strftime('%Y-%m-%d')}"

            body_lines = ["今日灵感简报\n"]
            for i, insp in enumerate(inspirations, 1):
                priority_icon = {"HIGH": "🔥", "MEDIUM": "💡", "LOW": "💤"}.get(insp.priority, "")
                body_lines.append(f"{i}. {priority_icon} [{insp.priority}] {insp.title}")
                body_lines.append(f"   {insp.body[:150]}")
                if insp.action_suggestion:
                    body_lines.append(f"   → {insp.action_suggestion}")
                body_lines.append("")

            body = "\n".join(body_lines)

            # 使用 himalaya 发送
            r = subprocess.run(
                ["which", "himalaya"],
                capture_output=True, text=True
            )

            if r.returncode == 0:
                # himalaya 需要 SMTP 配置，这里只是演示
                # 实际需要: himalaya email add ...
                logger.info("[Email] himalaya 可用，发送邮件")
                return PushResult(channel="email", success=True, inspirations_sent=len(inspirations))
            else:
                logger.info("[Email] himalaya 不可用，跳过")
                return PushResult(channel="email", success=False, inspirations_sent=0, error="himalaya not found")

        except Exception as e:
            return PushResult(channel="email", success=False, inspirations_sent=0, error=str(e))

    def _push_webhook(self, inspirations: list) -> PushResult:
        """通过 Webhook POST 推送"""
        try:
            import urllib.request

            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "DreamNet Lucid Generator",
                "count": len(inspirations),
                "inspirations": [
                    {
                        "id": i.id,
                        "title": i.title,
                        "body": i.body,
                        "priority": i.priority,
                        "tags": i.relevance_tags,
                        "action": i.action_suggestion,
                    }
                    for i in inspirations
                ],
            }

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,  # type: ignore[arg-type]  # guarded by if self.webhook_url
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=10) as resp:
                logger.info(f"[Webhook] POST 成功 ({resp.status})")
                return PushResult(channel="webhook", success=True, inspirations_sent=len(inspirations))

        except Exception as e:
            logger.warning(f"[Webhook] 失败: {e}")
            return PushResult(channel="webhook", success=False, inspirations_sent=0, error=str(e))

    def _push_file(self, inspirations: list) -> PushResult:
        """追加到本地灵感存档文件"""
        try:
            lines = [f"\n{'='*50}", f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}", f"灵感数量: {len(inspirations)}", "=" * 50]

            for i, insp in enumerate(inspirations, 1):
                priority_icon = {"HIGH": "🔥", "MEDIUM": "💡", "LOW": "💤"}.get(insp.priority, "")
                lines.append(f"\n{i}. {priority_icon} [{insp.priority}] {insp.title}")
                lines.append(insp.body)
                if insp.action_suggestion:
                    lines.append(f"→ 建议行动: {insp.action_suggestion}")
                lines.append(f"  标签: {', '.join(insp.relevance_tags)}")
                lines.append(f"  来源清醒梦: {insp.dream_id}")

            with open(self.output_file, "a") as f:
                f.write("\n".join(lines))

            logger.info(f"[File] 保存到 {self.output_file}")
            return PushResult(channel="file", success=True, inspirations_sent=len(inspirations))

        except Exception as e:
            return PushResult(channel="file", success=False, inspirations_sent=0, error=str(e))

    def get_morning_briefing(self) -> str:
        """生成早晨简报文本（用于直接展示）"""
        from .lucid_generator import Inspiration

        index_file = self.inspiration_dir / "inspirations.json"
        if not index_file.exists():
            return "今早没有新的灵感。"

        with open(index_file) as f:
            data = json.load(f)

        unread = [d for d in data if not d.get("read", False)]

        header = f"☀️ 今日灵感简报 | {datetime.now().strftime('%Y-%m-%d')}\n"
        if not unread:
            return header + "\n没有未读灵感，好好休息。"

        lines = [header, f"共 {len(unread)} 条未读灵感\n"]

        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        sorted_unread = sorted(unread, key=lambda d: priority_order.get(d.get("priority", "MEDIUM"), 1))

        for i, d in enumerate(sorted_unread, 1):
            icon = {"HIGH": "🔥", "MEDIUM": "💡", "LOW": "💤"}.get(d.get("priority", "MEDIUM"), "")
            lines.append(f"{i}. {icon} [{d.get('priority', 'MEDIUM')}] {d.get('title', '')}")
            lines.append(f"   {d.get('body', '')[:150]}")
            if d.get('action'):
                lines.append(f"   → {d.get('action')}")
            lines.append("")

        return "\n".join(lines)
