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
        
        # ── 输出发送给 LLM 的 prompt 内容 (让用户能看到 agent 发了什么!) ──
        print(f"\n{'┏'*60}")
        print(f"  [LLM Request] model={self.model}, temp={temperature}, max_tokens={max_tokens}")
        print(f"{'┏'*60}")
        for msg in messages:
            role = msg.get("role", "unknown")
            msg_content = msg.get("content", "")
            print(f"  [{role}] ({len(msg_content)} chars)")
            if len(msg_content) > 500:
                print(f"    {msg_content[:500]}")
                print(f"    ... (省略 {len(msg_content) - 500} 字符) ...")
            else:
                print(f"    {msg_content}")
        print(f"{'┏'*60}\n")
        logger.info(f"LLM request messages: {len(messages)} msgs, "
                    f"total {sum(len(m.get('content','')) for m in messages)} chars")

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
                
                # ── 输出 LLM 完整响应内容 (让用户能看到 LLM 到底说了什么!) ──
                if content:
                    # 输出到控制台 (用户可见)
                    print(f"\n{'━'*60}")
                    print(f"  [LLM Response] model={self.model}, {len(content)} chars")
                    print(f"{'━'*60}")
                    # 如果内容太长, 截断显示前 2000 字符 + 末尾摘要
                    if len(content) > 2000:
                        print(content[:2000])
                        print(f"\n  ... (省略 {len(content) - 2000} 字符) ...\n")
                        print(content[-500:])
                    else:
                        print(content)
                    print(f"{'━'*60}\n")
                    
                    # 输出到日志文件 (完整内容, 不截断)
                    logger.info(f"LLM raw response ({len(content)} chars):\n{content}")
                
                if content and len(content) > 10:
                    logger.info(f"LLM response OK ({len(content)} chars)")
                    return content
                else:
                    # 即使内容太短也输出, 让用户看到 LLM 返回了什么
                    print(f"\n{'━'*60}")
                    print(f"  [LLM Response — TOO SHORT] model={self.model}, {len(content) if content else 0} chars")
                    print(f"{'━'*60}")
                    print(content if content else "(empty)")
                    print(f"{'━'*60}\n")
                    logger.warning(f"LLM response too short: {content}")
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
                    print(f"\n{'✗'*30} LLM ERROR: Context too long {'✗'*30}\n")
                    logger.error("Context too long, need to truncate")
                    return None

                # 鉴权失败
                elif "auth" in error_str or "unauthorized" in error_str or "401" in error_str or "403" in error_str:
                    print(f"\n{'✗'*30} LLM AUTH FAILED {'✗'*30}\n  Error: {e}\n{'✗'*30}\n")
                    logger.error(f"LLM auth failed: {e}")
                    return None

                # 其他错误: 重试
                else:
                    print(f"\n  ⚠ LLM error (attempt {attempt}/{self.max_retries}): {e}\n")
                    logger.warning(f"LLM error (attempt {attempt}): {e}")
                    if attempt < self.max_retries:
                        time.sleep(2 ** attempt)
                    continue

        # ── 所有重试失败时输出到控制台 ──
        print(f"\n{'✗'*30} LLM CALL FAILED {'✗'*30}")
        print(f"  All {self.max_retries} attempts failed. Last error: {last_error}")
        print(f"{'✗'*30}\n")
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