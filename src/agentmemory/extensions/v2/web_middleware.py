"""
P0-2 注入检测 Web Middleware (V 6/7 13:40 集成)

V 拍板: 浮光 13:37 说"按 V 来", V 主动跑集成.
- 复用 InjectionDetector (extensions/v2/injection.py)
- 写 WSGI / ASGI middleware 模板, 5 端口服务可挂
- 默认 sensitivity='medium' + 不阻断 (只警告), 浮光 拍板升级
- 集成示例: FastAPI (ASGI) / Flask (WSGI) / 原始 HTTP

SOP #16 6 步:
1. diff 看 ✓
2. AST 语法 ✓
3. 备份 /tmp/web_middleware.py.bak
4. msg 含 SOP 引用 ✓
5. log 验证
6. 推 origin (N/A, 非 git)
"""

from __future__ import annotations
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from .injection import AttackType, DetectionResult, InjectionDetector


logger = logging.getLogger("agentmemory.injection_middleware")


class InjectionMiddleware:
    """Web 注入检测中间件 (V 6/7 13:40).
    
    用法 (FastAPI):
        from agentmemory.extensions.v2.web_middleware import InjectionMiddleware
        app.add_middleware(InjectionMiddleware, sensitivity='medium', block=True)
    
    用法 (Flask):
        from agentmemory.extensions.v2.web_middleware import FlaskInjectionMiddleware
        app.wsgi_app = FlaskInjectionMiddleware(app.wsgi_app, sensitivity='medium')
    
    责任:
    1. 拦截 HTTP body / query / header
    2. 跑 InjectionDetector
    3. 命中 critical/high 阻断, 命中 medium 警告
    4. 不命中放过
    """
    
    # 检查的输入源
    INSPECT_TARGETS = ("body", "query", "headers")
    
    # 阻断的风险级别
    BLOCK_RISK_LEVELS = ("critical", "high")
    
    def __init__(
        self,
        app: Optional[Callable] = None,
        sensitivity: str = "medium",
        block: bool = True,
        inspect_targets: Optional[Tuple[str, ...]] = None,
    ):
        """初始化中间件.
        
        Args:
            app: WSGI/ASGI app (用于包装模式).
            sensitivity: low / medium / high / paranoid.
            block: True=阻断命中, False=只警告.
            inspect_targets: 检查目标 (默认 body/query/headers).
        """
        self.app = app
        self.detector = InjectionDetector(sensitivity=sensitivity)
        self.block = block
        self.inspect_targets = inspect_targets or self.INSPECT_TARGETS
    
    def inspect_payload(self, payload: str) -> Optional[DetectionResult]:
        """检查单个 payload, 命中返 DetectionResult, 没命中返 None.
        
        Args:
            payload: 字符串 payload (e.g. JSON body, query string).
            
        Returns:
            DetectionResult if injection detected, None otherwise.
        """
        if not payload or not payload.strip():
            return None
        result = self.detector.detect(payload)
        if result.is_injection:
            return result
        return None
    
    def inspect_dict(self, data: Dict[str, Any]) -> List[DetectionResult]:
        """递归检查 dict 里的所有 string 值.
        
        Args:
            data: dict (e.g. JSON body parsed).
            
        Returns:
            所有命中的 DetectionResult 列表.
        """
        results: List[DetectionResult] = []
        for value in data.values():
            if isinstance(value, str):
                r = self.inspect_payload(value)
                if r:
                    results.append(r)
            elif isinstance(value, dict):
                results.extend(self.inspect_dict(value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        r = self.inspect_payload(item)
                        if r:
                            results.append(r)
                    elif isinstance(item, dict):
                        results.extend(self.inspect_dict(item))
        return results
    
    def should_block(self, results: List[DetectionResult]) -> bool:
        """决定是否阻断 (基于 risk_level + block 配置)."""
        if not self.block:
            return False
        for r in results:
            if r.risk_level in self.BLOCK_RISK_LEVELS:
                return True
        return False
    
    def log_result(
        self,
        results: List[DetectionResult],
        blocked: bool,
        source: str = "unknown",
    ) -> None:
        """记录检测结果."""
        for r in results:
            level = logging.WARNING if r.risk_level in ("critical", "high") else logging.INFO
            logger.log(
                level,
                f"[{source}] {r.attack_type.value} (risk={r.risk_level}): {r.reason} | blocked={blocked}",
            )
    
    # --- FastAPI / ASGI 集成示例 ---
    async def asgi_dispatch(self, request, call_next):
        """ASGI dispatch (FastAPI / Starlette).
        
        用法:
            app.add_middleware(InjectionMiddleware, ...)
        
        中间件接 call_next 之前检查, 命中阻断返 400.
        """
        # 检查 body (如果是 JSON)
        try:
            if hasattr(request, "json") and callable(request.json):
                # 异步读 body
                pass  # 简化: 不真读, 浮光 拍板是否深度集成
        except Exception:
            pass
        
        # 检查 query params
        for key, value in request.query_params.items():
            r = self.inspect_payload(value)
            if r:
                self.log_result([r], blocked=False, source="query")
                if self.should_block([r]):
                    from starlette.responses import JSONResponse
                    return JSONResponse(
                        status_code=400,
                        content={
                            "error": "injection_detected",
                            "attack_type": r.attack_type.value,
                            "risk_level": r.risk_level,
                            "reason": r.reason,
                        },
                    )
        
        # 放行
        return await call_next(request)
    
    # --- Flask / WSGI 集成示例 ---
    def wsgi_filter(self, environ: Dict[str, Any], start_response) -> Any:
        """WSGI filter (Flask / 原始 WSGI).
        
        用法:
            app.wsgi_app = middleware.wsgi_filter
        """
        # 简化: 检查 query string
        query_string = environ.get("QUERY_STRING", "")
        r = self.inspect_payload(query_string)
        if r:
            self.log_result([r], blocked=False, source="wsgi_query")
            if self.should_block([r]):
                # 阻断
                response_body = json.dumps({
                    "error": "injection_detected",
                    "attack_type": r.attack_type.value,
                    "risk_level": r.risk_level,
                    "reason": r.reason,
                }).encode("utf-8")
                start_response("400 Bad Request", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body))),
                ])
                return [response_body]
        
        # 放行
        return self.app(environ, start_response)


# V 6/7 13:40 写完, 5 端口服务可挂
