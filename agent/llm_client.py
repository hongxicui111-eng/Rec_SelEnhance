"""
LLM 客户端 — 带指数退避重试、超时保护、多种后端适配
"""
import time
import logging
from typing import Optional

logger = logging.getLogger("rec_self_evolve.llm")


class LLMClient:
    """
    LLM 调用客户端
    支持: OpenAI 兼容接口 (vLLM / Ollama / TGI)
    功能: 指数退避重试, 超时保护, 降级处理
    """

    def __init__(self, api_url: str, api_key: str = "EMPTY",
                 model: str = "Qwen2.5-72B-Instruct",
                 timeout: int = 120, max_retries: int = 3):
        self.api_url = api_url.rstrip("/") + "/v1" if "/v1" not in api_url else api_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.api_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._client

    def chat(self, messages: list, temperature: float = 0.7,
             max_tokens: int = 4096) -> Optional[str]:
        """
        调用 LLM 聊天接口, 带完整重试逻辑
        返回: 模型回复文本, 或 None (所有重试均失败)
        """
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"LLM call attempt {attempt}/{self.max_retries} "
                            f"(model={self.model})")
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = resp.choices[0].message.content
                if content and len(content) > 10:
                    logger.info(f"LLM response OK ({len(content)} chars)")
                    return content
                else:
                    logger.warning(f"LLM response too short ({len(content)} chars), retrying")
                    continue

            except ImportError:
                logger.error("openai package not installed. Run: pip install openai")
                return None

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # ---- 分类处理 ----

                # 超时: 指数退避
                if "timeout" in error_str or "timed out" in error_str:
                    wait = 2 ** attempt
                    logger.warning(f"LLM timeout (attempt {attempt}), waiting {wait}s...")
                    time.sleep(wait)
                    continue

                # 限流: 长等待
                elif "rate limit" in error_str or "429" in error_str:
                    wait = 30 * attempt
                    logger.warning(f"LLM rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue

                # 服务不可用
                elif "service unavailable" in error_str or "503" in error_str:
                    wait = 10 * attempt
                    logger.warning(f"LLM service unavailable, waiting {wait}s...")
                    time.sleep(wait)
                    continue

                # 模型过载
                elif "overloaded" in error_str or "capacity" in error_str:
                    wait = 15 * attempt
                    logger.warning(f"LLM overloaded, waiting {wait}s...")
                    time.sleep(wait)
                    continue

                # 上下文过长
                elif "maximum context length" in error_str or "too many tokens" in error_str:
                    logger.error("Context too long, need to truncate")
                    return None

                # 鉴权失败
                elif "auth" in error_str or "unauthorized" in error_str or "401" in error_str or "403" in error_str:
                    logger.error(f"LLM auth failed: {e}")
                    return None

                # 其他错误: 重试
                else:
                    logger.warning(f"LLM error (attempt {attempt}): {e}")
                    if attempt < self.max_retries:
                        time.sleep(2 ** attempt)
                    continue

        logger.error(f"LLM failed after {self.max_retries} attempts. Last error: {last_error}")
        return None

    def check_health(self) -> bool:
        """检查 LLM 服务是否可用"""
        try:
            import requests
            resp = requests.get(
                self.api_url.replace("/v1", "") + "/health",
                timeout=5
            )
            return resp.status_code == 200
        except Exception:
            try:
                result = self.chat(
                    messages=[{"role": "user", "content": "ping"}],
                    temperature=0.1, max_tokens=10
                )
                return result is not None
            except Exception:
                return False