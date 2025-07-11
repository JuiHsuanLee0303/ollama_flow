"""
基本使用範例
"""

from ollama_flow import OllamaClient, ChatMessage


def basic_generate_example():
    """基本生成範例"""
    print("=== 基本生成範例 ===")
    
    client = OllamaClient()
    
    response = client.generate(
        model="qwen3:4b-q4_K_M",
        prompt="什麼是 Python？請簡短回答。",
        stream=False
    )
    
    print(f"模型：{response.model}")
    print(f"回應：{response.response}")
    print(f"生成時間：{response.eval_duration / 1e9:.2f}秒")


def basic_chat_example():
    """基本聊天範例"""
    print("\n=== 基本聊天範例 ===")
    
    client = OllamaClient()
    
    messages = [
        ChatMessage(role="system", content="你是一個有用的助手。"),
        ChatMessage(role="user", content="請介紹一下自己。")
    ]
    
    response = client.chat(
        model="qwen3:4b-q4_K_M",
        messages=messages,
        stream=False
    )
    
    print(f"模型：{response.model}")
    print(f"回應：{response.message.content}")


def basic_embed_example():
    """基本嵌入範例"""
    print("\n=== 基本嵌入範例 ===")
    
    client = OllamaClient()
    
    response = client.embed(
        model="bge-m3:latest",
        input="這是一段測試文本用於生成嵌入向量。"
    )
    
    print(f"模型：{response.model}")
    print(f"嵌入維度：{len(response.embeddings[0])}")
    print(f"嵌入前5個值：{response.embeddings[0][:5]}")


if __name__ == "__main__":
    basic_generate_example()
    basic_chat_example()
    basic_embed_example() 