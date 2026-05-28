"""
LLM 客户端 — 带指数退避重试、超时保护、多种后端适配
"""
import time
import logging
from typing import Optional
import re

logger = logging.getLogger("rec_self_evolve.llm")


class LLMClient:
    """
    LLM 调用客户端
    支持: OpenAI 兼容接口 (vLLM / Ollama / TGI)
    功能: 指数退避重试, 超时保护, 降级处理
    """

    def __init__(self, api_url: str, api_key: str = "EMPTY",
                 model: str = "Qwen2.5-72B-Instruct",
                 timeout: int = 120, max_retries: int = 3,
                 max_context_tokens: int = 32768,
                 prompt_safety_ratio: float = 0.75):
        self.api_url = api_url.rstrip("/") + "/v1" if "/v1" not in api_url else api_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_context_tokens = max_context_tokens
        self.prompt_safety_ratio = max(0.1, min(prompt_safety_ratio, 0.95))
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

    def _estimate_tokens(self, text: str) -> int:
        """粗略估算 token（中文/英文混合场景下取保守值）"""
        if not text:
            return 0
        return max(1, len(text) // 3)

    def _estimate_messages_tokens(self, messages: list) -> int:
        total = 0
        for m in messages:
            total += self._estimate_tokens(m.get("content", "")) + 8
        return total

    def _truncate_text(self, text: str, keep_chars: int) -> str:
        if len(text) <= keep_chars:
            return text
        if keep_chars <= 120:
            return text[:max(0, keep_chars)]
        head = int(keep_chars * 0.7)
        tail = keep_chars - head - 30
        return text[:head] + "\n\n... [TRUNCATED] ...\n\n" + text[-max(0, tail):]

    def _compress_message_content(self, content: str, max_chars: int) -> str:
        """优先压缩大代码块和大 JSON，再做通用截断"""
        if len(content) <= max_chars:
            return content

        # 1) 先压 markdown 代码块
        def _code_block_repl(match):
            body = match.group(0)
            if len(body) <= 1200:
                return body
            return self._truncate_text(body, 1200)

        compressed = re.sub(r"```[\s\S]*?```", _code_block_repl, content)
        if len(compressed) <= max_chars:
            return compressed

        # 2) 再压长 JSON 片段
        if "{" in compressed and "}" in compressed and len(compressed) > max_chars:
            compressed = self._truncate_text(compressed, int(max_chars * 0.9))

        # 3) 最后兜底截断
        if len(compressed) > max_chars:
            compressed = self._truncate_text(compressed, max_chars)

        return compressed

    def _fit_messages_to_context(self, messages: list, max_tokens: int) -> list:
        """按上下文预算裁剪消息内容，优先保留 system 和最新 user 信息"""
        if not messages:
            return messages

        prompt_budget_tokens = int(self.max_context_tokens * self.prompt_safety_ratio)
        prompt_budget_tokens = max(1024, prompt_budget_tokens - max_tokens)

        est = self._estimate_messages_tokens(messages)
        if est <= prompt_budget_tokens:
            return messages

        logger.warning(
            f"Prompt too long before send: est_tokens={est}, budget={prompt_budget_tokens}. Start compressing."
        )

        fitted = [dict(m) for m in messages]

        # 第一轮：压缩 user 消息（从最老到最新）
        for i, m in enumerate(fitted):
            if m.get("role") != "user":
                continue
            content = m.get("content", "")
            if len(content) > 4000:
                m["content"] = self._compress_message_content(content, 4000)
                fitted[i] = m
            if self._estimate_messages_tokens(fitted) <= prompt_budget_tokens:
                return fitted

        # 第二轮：压缩 system 消息（保守压缩）
        for i, m in enumerate(fitted):
            if m.get("role") != "system":
                continue
            content = m.get("content", "")
            if len(content) > 3000:
                m["content"] = self._compress_message_content(content, 3000)
                fitted[i] = m
            if self._estimate_messages_tokens(fitted) <= prompt_budget_tokens:
                return fitted

        # 第三轮：如果仍超预算，保留最后两条消息 + system
        system_msgs = [m for m in fitted if m.get("role") == "system"]
        non_system = [m for m in fitted if m.get("role") != "system"]
        tail = non_system[-2:] if len(non_system) > 2 else non_system
        fitted = system_msgs[:1] + tail if system_msgs else tail

        # 第四轮：最后兜底强截断每条消息
        hard_cap_chars = 2400
        for i, m in enumerate(fitted):
            m["content"] = self._compress_message_content(m.get("content", ""), hard_cap_chars)
            fitted[i] = m

        return fitted

    def chat(self, messages: list, temperature: float = 0.7,
             max_tokens: int = 4096, suppress_response_log: bool = False) -> Optional[str]:
        """
        调用 LLM 聊天接口, 带完整重试逻辑
        返回: 模型回复文本, 或 None (所有重试均失败)
        
        Args:
            suppress_response_log: 如果为 True, 不输出响应日志 (用于代码生成等场景)
        """
        last_error = None
        messages = self._fit_messages_to_context(messages, max_tokens=max_tokens)
        est_tokens = self._estimate_messages_tokens(messages)
        logger.info(f"LLM prompt estimated tokens after fit: {est_tokens}")
        
        # ── 输出发送给 LLM 的 prompt 内容 (让用户能看到 agent 发了什么!) ──
        print(f"\n{'┏'*60}")
        print(f"  [LLM Request] model={self.model}, temp={temperature}, max_tokens={max_tokens}")
        print(f"{'┏'*60}")
        for msg in messages:
            role = msg.get("role", "unknown")
            msg_content = msg.get("content", "")
            print(f"  [{role}] ({len(msg_content)} chars)")
            if len(msg_content) > 10000:
                print(f"    {msg_content[:10000]}")
                print(f"    ... (省略 {len(msg_content) - 10000} 字符) ...")
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
                
                # ── 输出 LLM 响应到日志 (完整内容, 不截断) ──
                # 如果 suppress_response_log=True, 则跳过日志输出 (用于代码生成等场景)
                if not suppress_response_log:
                    logger.info(f"LLM raw response ({len(content)} chars):\n{content}")
                else:
                    logger.info(f"LLM response ({len(content)} chars) [suppressed]")
                
                if content and len(content.strip()) > 0:
                    logger.info(f"LLM response OK ({len(content)} chars)")
                    return content
                else:
                    # 空响应输出到日志
                    logger.warning(f"LLM response empty or too short: '{content}'")
                    # 对空回复重试, 但短回复(有内容)直接返回
                    if not content or len(content.strip()) == 0:
                        continue
                    else:
                        # 短但非空回复 — 直接返回, 让调用方决定是否有效
                        logger.info(f"LLM response short but non-empty ({len(content)} chars), returning it")
                        return content

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
                    logger.error("Context too long, retry with more aggressive compression")
                    messages = self._fit_messages_to_context(messages, max_tokens=max_tokens)
                    # 进一步降低输出长度，换取更多输入空间
                    max_tokens = max(512, int(max_tokens * 0.8))
                    if attempt < self.max_retries:
                        continue
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

    async def async_chat(self, prompt_or_messages, temperature: float = 0.7,
                         max_tokens: int = 4096) -> Optional[str]:
        """
        异步包装版本 — 支持传入字符串 prompt 或 messages 列表
        
        Args:
            prompt_or_messages: 字符串 prompt (会自动转为 messages 列表) 或 messages 列表
            temperature: 采样温度
            max_tokens: 最大输出 token 数
            
        Returns:
            模型回复文本, 或 None (所有重试均失败)
        """
        import asyncio
        
        # 如果传入字符串, 自动转为 messages 列表
        if isinstance(prompt_or_messages, str):
            messages = [{"role": "user", "content": prompt_or_messages}]
        else:
            messages = prompt_or_messages
        
        return await asyncio.to_thread(
            self.chat, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )

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