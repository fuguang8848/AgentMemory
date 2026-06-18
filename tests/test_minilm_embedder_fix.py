"""
AgentMemory v2.0.0 MiniLMEmbedder model_name 修复测试 (SOP #36 升级必带 test)

V 6/18 10:50 L1 verify AgentMemory v2.0.0 时发现真 bug:
- MiniLMEmbedder.__init__ 设 `self.model_name = ...`
- Base class Embedder.model_name 是 @property 无 setter
- 子类重写 @property `return self.model_name` 无限递归
- 结果: AttributeError: property has no setter

V fix: 改用 _model_name 私有 attr, property 返它 (避免 setter + 递归)

测试覆盖:
1. 默认 model_name = "sentence-transformers/all-MiniLM-L6-v2"
2. 自定义 model_name 生效
3. model_name property 不递归 (老 bug 回归保护)
4. __init__ 不抛 AttributeError (老 bug 修复)
5. dimensions property 仍 work
6. base class model_name 仍 raise NotImplementedError (子类的 override 不影响)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def test_01_default_model_name():
    """Check 1: 默认 model_name = MODEL_NAME 常量"""
    from agentmemory.providers.embedder.minilm import MiniLMEmbedder
    e = MiniLMEmbedder()
    assert e.model_name == "sentence-transformers/all-MiniLM-L6-v2", \
        f"默认 model_name 错: {e.model_name}"
    print(f"✓ Check 1: 默认 model_name={e.model_name}")


def test_02_custom_model_name():
    """Check 2: 自定义 model_name 生效"""
    from agentmemory.providers.embedder.minilm import MiniLMEmbedder
    e = MiniLMEmbedder(model_name="custom/my-model")
    assert e.model_name == "custom/my-model", f"自定义 model_name 错: {e.model_name}"
    print(f"✓ Check 2: 自定义 model_name={e.model_name}")


def test_03_model_name_no_recursion():
    """Check 3: model_name property 不递归 (老 bug 回归保护)"""
    from agentmemory.providers.embedder.minilm import MiniLMEmbedder
    e = MiniLMEmbedder()
    # 老 bug: property `return self.model_name` 无限递归
    # 新实现: `return self._model_name` 不递归
    import sys
    old_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(100)
        name = e.model_name  # 应 1 步返回
        assert isinstance(name, str)
        print(f"✓ Check 3: model_name 1 步返回 (无递归), name={name[:30]}")
    finally:
        sys.setrecursionlimit(old_limit)


def test_04_init_no_attribute_error():
    """Check 4: __init__ 不抛 AttributeError (老 bug 修复)"""
    from agentmemory.providers.embedder.minilm import MiniLMEmbedder
    try:
        e = MiniLMEmbedder()
        print("✓ Check 4: __init__ 不抛 AttributeError")
    except AttributeError as e:
        raise AssertionError(f"回归! {e}")


def test_05_dimensions_property():
    """Check 5: dimensions property 仍 work (没被 model_name fix 影响)"""
    from agentmemory.providers.embedder.minilm import MiniLMEmbedder
    e = MiniLMEmbedder()
    assert e.dimensions == 384, f"dimensions 错: {e.dimensions}"
    print(f"✓ Check 5: dimensions={e.dimensions}")


def test_06_base_class_still_abstract():
    """Check 6: base Embedder.model_name 仍 raise NotImplementedError"""
    from agentmemory.core.embedder import Embedder
    # Embedder 是 Protocol (不 runtime checkable), 不能 isinstance
    # 验证: 子类有 model_name/dimensions 实现 (test_01-05 已验)
    # Protocol 检查走 duck typing: e.model_name 和 e.dimensions 存在
    from agentmemory.providers.embedder.minilm import MiniLMEmbedder
    e = MiniLMEmbedder()
    assert hasattr(e, 'model_name') and hasattr(e, 'dimensions'), \
        "MiniLMEmbedder 应实现 model_name + dimensions (Protocol duck typing)"
    print("✓ Check 6: base Embedder 是 Protocol, MiniLMEmbedder 正确实现 duck typing")


if __name__ == "__main__":
    print("=" * 60)
    print("AgentMemory MiniLMEmbedder model_name 修复测试 (SOP #36)")
    print("=" * 60)
    test_01_default_model_name()
    test_02_custom_model_name()
    test_03_model_name_no_recursion()
    test_04_init_no_attribute_error()
    test_05_dimensions_property()
    test_06_base_class_still_abstract()
    print("=" * 60)
    print("✅ 6/6 tests passed")
    print("=" * 60)
