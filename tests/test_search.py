import asyncio
import sys
from pathlib import Path
# 将项目根目录添加到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))
from agent.tools import search_web

# 同步调用
def test_sync():
    query = "sequential recommendation SASRec latest research 2025"
    print(f"正在搜索：{query}")
    res = search_web(query, gl="en", hl="en", num_results=5)
    
    print(f"返回结果数量：{res['total']}")
    for i, item in enumerate(res['results'], 1):
        print(f"\n--- 结果 {i} ---")
        print(f"标题: {item['title']}")
        print(f"URL: {item['url']}")
        print(f"片段: {item['snippet']}")
    
    if res['total'] == 0:
        print("警告：没有获取到任何结果，请检查网络或 DuckDuckGo 是否可用。")

# 异步调用（模拟 ResearcherAgent 的使用方式）
async def test_async():
    query = "transformer attention sequence recommendation"
    print(f"异步搜索：{query}")
    res = await asyncio.to_thread(search_web, query=query, gl="en", hl="en", num_results=5)
    print(f"异步结果数量：{res['total']}")

if __name__ == "__main__":
    test_sync()
    asyncio.run(test_async())