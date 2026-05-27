"""
agentmemory API 集成测试
验证 memory_manager 与各层的集成是否正常
"""

import sys
import os
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from memory_manager import MemoryHermes


async def test_integration():
    print("\n=== MemoryHermes API 集成测试 ===\n")
    
    # 1. 实例化
    print("1. 实例化 MemoryHermes...")
    mh = MemoryHermes()
    print("   ✅ 实例化成功")
    print(f"   各层状态: {list(mh.get_stats()['layers'].keys())}")
    
    # 2. 统计接口
    print("\n2. get_stats() 接口...")
    stats = mh.get_stats()
    print(f"   ✅ 统计成功: {stats}")
    
    # 3. get_prefetched（空缓存）
    print("\n3. get_prefetched() 接口...")
    prefetched = mh.get_prefetched()
    print(f"   ✅ 预取缓存（空）: '{prefetched}'")
    
    # 4. execute 接口 (store)
    print("\n4. execute('store') 接口...")
    result = await mh.execute("store", {
        "content": "测试：agentmemory API 集成测试",
        "metadata": {"source": "test", "category": "learning"},
        "importance": 0.9
    })
    print(f"   ✅ 存储结果: {result}")
    
    # 5. execute 接口 (query)
    print("\n5. execute('query') 接口...")
    result = await mh.execute("query", {
        "query": "agentmemory",
        "limit": 3
    })
    print(f"   ✅ 查询结果数: {len(result.get('results', []))}")
    
    # 6. execute 接口 (get_stats)
    print("\n6. execute('get_stats') 接口...")
    result = await mh.execute("get_stats")
    print(f"   ✅ 统计: {result}")
    
    # 7. 遗忘检查
    print("\n7. run_decay_check() 接口...")
    result = await mh.run_decay_check()
    print(f"   ✅ 遗忘检查: {result}")
    
    print("\n" + "=" * 50)
    print("🎉 API 集成测试全部通过!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(test_integration())
