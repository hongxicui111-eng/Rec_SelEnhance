"""
Tools 模块 - 提供网络搜索等工具函数

功能：
- search_web: 网络搜索功能（基于 DuckDuckGo）
"""

import logging
import urllib.parse
from typing import Dict, List, Any, Optional

logger = logging.getLogger("rec_self_evolve.tools")

# 尝试导入必要的库
try:
    import requests
    from bs4 import BeautifulSoup
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    logger.warning("requests or beautifulsoup4 not installed. Install with: pip install requests beautifulsoup4")

# 广告/无效域名黑名单
AD_DOMAINS = {
    "duckduckgo.com", "bing.com", "google.com", "yahoo.com",
    "udemy.com", "coursera.org", "skillshare.com", "udacity.com",
    "youtube.com", "amazon.com", "wikipedia.org",  # Wikipedia 通常不是最佳研究来源
}


def _is_valid_url(url: str) -> bool:
    """检查 URL 是否有效（过滤广告和无效链接）"""
    if not url or not url.startswith("http"):
        return False
    
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        
        # 移除 www. 前缀
        if domain.startswith("www."):
            domain = domain[4:]
        
        # 检查是否是广告/无效域名
        for ad_domain in AD_DOMAINS:
            if domain == ad_domain or domain.endswith(f".{ad_domain}"):
                return False
        
        # 检查是否是 DuckDuckGo 重定向
        if "duckduckgo.com" in domain:
            return False
            
        return True
    except Exception:
        return False


def search_web(
    query: str,
    gl: str = "en",
    hl: str = "en",
    num_results: int = 10,
    timeout: int = 10,
) -> Dict[str, Any]:
    """
    网络搜索功能 - 使用 DuckDuckGo HTML 搜索
    
    Args:
        query: 搜索关键词
        gl: 国家代码 (如 "en", "cn")
        hl: 语言代码 (如 "en", "zh-CN")
        num_results: 返回结果数量
        timeout: 请求超时时间（秒）
        
    Returns:
        {
            "results": [
                {
                    "title": str,
                    "url": str,
                    "snippet": str,
                },
                ...
            ],
            "query": str,
            "total": int
        }
    """
    if not DEPENDENCIES_AVAILABLE:
        logger.warning("Dependencies not available, returning empty results")
        return {"results": [], "query": query, "total": 0}
    
    # DuckDuckGo HTML 搜索端点
    url = "https://html.duckduckgo.com/html/"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": f"{hl},{gl};q=0.9,en-US;q=0.8",
    }
    
    data = {
        "q": query,
        "b": "",  # 起始位置
    }
    
    try:
        response = requests.post(
            url,
            data=data,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        results = []
        # DuckDuckGo HTML 结果在 result 类的 a 标签中
        for result in soup.select(".result"):
            try:
                link_elem = result.select_one(".result__a")
                snippet_elem = result.select_one(".result__snippet")
                
                if link_elem:
                    title = link_elem.get_text(strip=True)
                    url = link_elem.get("href", "")
                    
                    # 清理 URL (DuckDuckGo 使用重定向)
                    if "uddg=" in url:
                        # 提取真实 URL
                        try:
                            parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                            if "uddg" in parsed:
                                url = parsed["uddg"][0]
                        except Exception:
                            pass
                    
                    # 过滤无效 URL
                    if not _is_valid_url(url):
                        continue
                    
                    snippet = ""
                    if snippet_elem:
                        snippet = snippet_elem.get_text(strip=True)
                    
                    if title and url:
                        results.append({
                            "title": title,
                            "url": url,
                            "snippet": snippet,
                        })
                        
                        if len(results) >= num_results:
                            break
            except Exception as e:
                logger.debug(f"Failed to parse result: {e}")
                continue
        
        logger.info(f"Search completed for query: {query}, found {len(results)} results")
        return {
            "results": results,
            "query": query,
            "total": len(results),
        }
        
    except requests.exceptions.Timeout:
        logger.warning(f"Search timeout for query: {query}")
        return {"results": [], "query": query, "total": 0}
    except requests.exceptions.RequestException as e:
        logger.warning(f"Search request failed for query '{query}': {e}")
        return {"results": [], "query": query, "total": 0}
    except Exception as e:
        logger.warning(f"Search unexpected error for query '{query}': {e}")
        return {"results": [], "query": query, "total": 0}


async def async_search_web(
    search_web,
    query: str,
    gl: str = "en",
    hl: str = "en",
    num_results: int = 10,
    timeout: int = 10,
) -> Dict[str, Any]:
    """
    异步版本的搜索（如果需要）
    
    Args:
        search_web: 同步搜索函数
        其他参数同 search_web
        
    Returns:
        同 search_web
    """
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, search_web, query, gl, hl, num_results, timeout)


__all__ = ["search_web"]
