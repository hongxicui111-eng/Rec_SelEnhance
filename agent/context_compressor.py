"""
Context Compressor — LLM 语义上下文压缩器

升级点:
1. 分层压缩模板（journal/history/rollback/general）
2. 压缩结果缓存（hash key + TTL + 落盘）
3. 分块压缩 + 二次汇总 + 失败回退
"""

from typing import List, Dict, Any
import logging
import os
import json
import hashlib
import time

logger = logging.getLogger("rec_self_evolve.context_compressor")


class LLMContextCompressor:
    """基于现有 LLMClient 的上下文压缩器"""

    def __init__(self, llm_client,
                 enable_cache: bool = True,
                 cache_ttl_seconds: int = 86400,
                 cache_path: str = "logs/context_compression_cache.json"):
        self.llm = llm_client
        self.enable_cache = enable_cache
        self.cache_ttl_seconds = max(60, int(cache_ttl_seconds))
        self.cache_path = cache_path
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_dirty = False
        self._load_cache()

    def _load_cache(self):
        if not self.enable_cache:
            return
        try:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._cache = data
                    self._cleanup_expired_cache()
                    logger.info(f"Context compressor cache loaded: {len(self._cache)} entries")
        except Exception as e:
            logger.warning(f"Load compression cache failed: {e}")

    def _save_cache(self):
        if not self.enable_cache or not self._cache_dirty:
            return
        try:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False)
            self._cache_dirty = False
        except Exception as e:
            logger.warning(f"Save compression cache failed: {e}")

    def _cleanup_expired_cache(self):
        now = int(time.time())
        expired = []
        for k, v in self._cache.items():
            ts = int(v.get("ts", 0))
            if now - ts > self.cache_ttl_seconds:
                expired.append(k)
        if expired:
            for k in expired:
                self._cache.pop(k, None)
            self._cache_dirty = True

    @staticmethod
    def _hash_key(text: str, target_chars: int, section_name: str, profile: str) -> str:
        raw = f"{section_name}|{profile}|{target_chars}|{text}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _chunk_text(text: str, chunk_chars: int) -> List[str]:
        if not text:
            return []
        if len(text) <= chunk_chars:
            return [text]

        chunks = []
        i = 0
        n = len(text)
        while i < n:
            end = min(i + chunk_chars, n)
            probe = text[i:end]
            split_pos = probe.rfind("\n\n")
            if split_pos > int(chunk_chars * 0.5):
                end = i + split_pos
            if end <= i:
                end = min(i + chunk_chars, n)
            chunks.append(text[i:end])
            i = end
        return chunks

    @staticmethod
    def _profile_prompt(profile: str) -> str:
        prompts = {
            "journal": (
                "你在压缩实验日志。必须保留: 迭代轮次、status、关键指标变化、"
                "失败原因、结构修改与回滚信息。按'事实->变化->结论'组织。"
            ),
            "history": (
                "你在压缩结构修改历史。必须保留: 修改动作、目标文件/函数、"
                "效果(正/负/中性)、回滚黑名单、不可重复尝试的约束。"
            ),
            "rollback": (
                "你在压缩回滚警告。必须保留: 被回滚的修改、回滚原因、"
                "禁止重复的具体模式。不要扩写。"
            ),
            "general": (
                "你在压缩通用上下文。保留关键事实、参数、文件与函数名、"
                "错误根因和可执行约束。"
            ),
        }
        return prompts.get(profile, prompts["general"])

    def _compress_chunk(self, chunk: str, target_chars: int, profile: str) -> str:
        base = self._profile_prompt(profile)
        prompt = (
            f"{base}\n"
            "请将下面内容压缩为结构化摘要，严格保留关键约束，不要编造。\n"
            f"目标长度: 不超过 {target_chars} 字符。\n\n"
            "待压缩内容:\n"
            f"{chunk}"
        )

        resp = self.llm.chat(
            messages=[
                {"role": "system", "content": "你是精确的信息压缩器，不能遗漏关键约束。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        if not resp:
            return ""
        return resp.strip()

    def _summarize_merged(self, merged: str, target_chars: int, profile: str) -> str:
        base = self._profile_prompt(profile)
        final_prompt = (
            f"{base}\n"
            "请将以下分块摘要合并为一份最终摘要，要求:\n"
            "1) 去重\n"
            "2) 保留关键约束与因果关系\n"
            "3) 保留涉及的文件/函数/参数/指标\n"
            "4) 输出纯文本\n\n"
            f"最终长度不超过 {target_chars} 字符。\n\n"
            f"分块摘要:\n{merged}"
        )

        final_resp = self.llm.chat(
            messages=[
                {"role": "system", "content": "你是摘要整合器，输出高密度关键信息。"},
                {"role": "user", "content": final_prompt},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        return (final_resp or "").strip()

    def compress_text(self, text: str, chunk_chars: int = 5000,
                      target_chars: int = 3500,
                      section_name: str = "context",
                      profile: str = "general") -> str:
        """压缩任意长文本：分块压缩 + 汇总 + 缓存。"""
        if not text:
            return ""
        if len(text) <= target_chars:
            return text

        profile = profile or "general"

        # 缓存命中
        cache_key = self._hash_key(text, target_chars, section_name, profile)
        if self.enable_cache:
            item = self._cache.get(cache_key)
            if item:
                ts = int(item.get("ts", 0))
                if int(time.time()) - ts <= self.cache_ttl_seconds:
                    cached = item.get("value", "")
                    if cached:
                        logger.info(f"Semantic compression cache hit: section={section_name}, profile={profile}")
                        return cached

        chunks = self._chunk_text(text, chunk_chars=chunk_chars)
        if not chunks:
            return text

        logger.info(
            f"Semantic compression start: section={section_name}, profile={profile}, len={len(text)}, chunks={len(chunks)}"
        )

        compressed_chunks = []
        per_chunk_target = max(600, int(target_chars * 0.6))

        for idx, ch in enumerate(chunks, start=1):
            c = self._compress_chunk(ch, target_chars=per_chunk_target, profile=profile)
            if not c:
                logger.warning(f"Chunk compression failed: section={section_name}, idx={idx}")
                continue
            compressed_chunks.append(f"[Chunk {idx}]\n{c}")

        if not compressed_chunks:
            logger.warning(f"Semantic compression failed, fallback raw: section={section_name}")
            return text

        merged = "\n\n".join(compressed_chunks)
        if len(compressed_chunks) == 1 and len(merged) <= target_chars:
            out = merged
        else:
            out = self._summarize_merged(merged, target_chars=target_chars, profile=profile)
            if not out:
                out = merged[:target_chars]

        if len(out) > target_chars:
            out = out[:target_chars]

        # 写入缓存
        if self.enable_cache and out:
            self._cache[cache_key] = {
                "value": out,
                "ts": int(time.time()),
            }
            self._cache_dirty = True
            self._cleanup_expired_cache()
            self._save_cache()

        return out
