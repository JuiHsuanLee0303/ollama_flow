"""
模型驗證功能演示
"""

from ollama_flow import OllamaClient, ChatMessage


def test_model_validation():
    """測試模型驗證功能"""
    print("=== 模型驗證功能演示 ===")
    
    # 建立開啟模型檢查的客戶端
    client = OllamaClient(check_models=True)
    
    # 1. 顯示可用模型
    print("\n1. 獲取可用模型列表：")
    try:
        models = client.list_models()
        print(f"可用模型：{models}")
    except Exception as e:
        print(f"獲取模型列表失敗：{e}")
        return
    
    # 2. 使用存在的模型
    print("\n2. 使用存在的模型：")
    if models:
        valid_model = models[0]
        print(f"使用模型：{valid_model}")
        
        try:
            response = client.generate(
                model=valid_model,
                prompt="Hello!",
                stream=False
            )
            print(f"成功生成：{response.response[:100]}...")
        except Exception as e:
            print(f"生成失敗：{e}")
    
    # 3. 使用不存在的模型
    print("\n3. 使用不存在的模型：")
    invalid_model = "nonexistent-model:latest"
    print(f"嘗試使用不存在的模型：{invalid_model}")
    
    try:
        response = client.generate(
            model=invalid_model,
            prompt="Hello!",
            stream=False
        )
        print("意外成功！")
    except ValueError as e:
        print(f"預期的模型驗證錯誤：{e}")
    except Exception as e:
        print(f"其他錯誤：{e}")


def test_model_validation_disabled():
    """測試關閉模型驗證功能"""
    print("\n=== 關閉模型驗證功能演示 ===")
    
    # 建立關閉模型檢查的客戶端
    client = OllamaClient(check_models=False)
    
    invalid_model = "nonexistent-model:latest"
    print(f"關閉模型檢查，嘗試使用不存在的模型：{invalid_model}")
    
    try:
        response = client.generate(
            model=invalid_model,
            prompt="Hello!",
            stream=False
        )
        print("意外成功！")
    except ValueError as e:
        print(f"模型驗證錯誤（不應該出現）：{e}")
    except Exception as e:
        print(f"預期的API錯誤：{e}")


def test_cache_functionality():
    """測試緩存功能"""
    print("\n=== 緩存功能演示 ===")
    
    client = OllamaClient()
    
    print("第一次獲取模型列表（會從服務器獲取）：")
    try:
        models1 = client.list_models()
        print(f"獲取到 {len(models1)} 個模型")
    except Exception as e:
        print(f"獲取失敗：{e}")
        return
    
    print("第二次獲取模型列表（使用緩存）：")
    try:
        models2 = client.list_models()
        print(f"從緩存獲取到 {len(models2)} 個模型")
    except Exception as e:
        print(f"獲取失敗：{e}")
        return
    
    print("刷新緩存：")
    try:
        models3 = client.refresh_models_cache()
        print(f"刷新後獲取到 {len(models3)} 個模型")
    except Exception as e:
        print(f"刷新失敗：{e}")


def test_all_apis():
    """測試所有 API 的模型驗證"""
    print("\n=== 所有 API 模型驗證演示 ===")
    
    client = OllamaClient(check_models=True)
    
    # 獲取可用模型
    try:
        models = client.list_models()
        if not models:
            print("沒有可用模型")
            return
        
        valid_model = models[0]
        print(f"使用模型：{valid_model}")
        
        # 測試 generate API
        print("\n測試 generate API：")
        try:
            response = client.generate(
                model=valid_model,
                prompt="Say hello",
                stream=False
            )
            print("✓ generate API 模型驗證成功")
        except Exception as e:
            print(f"✗ generate API 失敗：{e}")
        
        # 測試 chat API
        print("\n測試 chat API：")
        try:
            messages = [ChatMessage(role="user", content="Hello")]
            response = client.chat(
                model=valid_model,
                messages=messages,
                stream=False
            )
            print("✓ chat API 模型驗證成功")
        except Exception as e:
            print(f"✗ chat API 失敗：{e}")
        
        # 測試 embed API
        print("\n測試 embed API：")
        # 尋找嵌入模型
        embed_models = [m for m in models if 'embed' in m.lower() or 'minilm' in m.lower() or 'bge' in m.lower()]
        if embed_models:
            embed_model = embed_models[0]
            print(f"使用嵌入模型：{embed_model}")
            try:
                response = client.embed(
                    model=embed_model,
                    input="Test text"
                )
                print("✓ embed API 模型驗證成功")
            except Exception as e:
                print(f"✗ embed API 失敗：{e}")
        else:
            print("沒有找到嵌入模型")
    
    except Exception as e:
        print(f"獲取模型列表失敗：{e}")


if __name__ == "__main__":
    test_model_validation()
    test_model_validation_disabled()
    test_cache_functionality()
    test_all_apis() 