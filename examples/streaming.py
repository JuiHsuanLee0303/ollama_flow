"""
串流模式範例
"""

from ollama_flow import OllamaClient, ChatMessage
import time


def generate_streaming_example():
    """生成串流範例"""
    print("=== 生成串流範例 ===")
    
    client = OllamaClient()
    
    print("正在生成回應...")
    print("回應內容：", end="", flush=True)
    
    response_stream = client.generate(
        model="llama3.2",
        prompt="請詳細解釋什麼是人工智慧，並舉例說明其應用領域。",
        stream=True
    )
    
    full_response = ""
    for chunk in response_stream:
        if chunk.get("done", False):
            print(f"\n\n=== 統計資訊 ===")
            print(f"總時間：{chunk.get('total_duration', 0) / 1e9:.2f}秒")
            print(f"載入時間：{chunk.get('load_duration', 0) / 1e9:.2f}秒")
            print(f"生成時間：{chunk.get('eval_duration', 0) / 1e9:.2f}秒")
            print(f"生成字元數：{chunk.get('eval_count', 0)}")
            if chunk.get('eval_count', 0) > 0 and chunk.get('eval_duration', 0) > 0:
                tokens_per_sec = chunk.get('eval_count', 0) / (chunk.get('eval_duration', 0) / 1e9)
                print(f"生成速度：{tokens_per_sec:.2f} tokens/秒")
            break
        else:
            response_text = chunk.get("response", "")
            print(response_text, end="", flush=True)
            full_response += response_text
    
    print(f"\n完整回應長度：{len(full_response)} 字元")


def chat_streaming_example():
    """聊天串流範例"""
    print("\n=== 聊天串流範例 ===")
    
    client = OllamaClient()
    
    messages = [
        ChatMessage(role="system", content="你是一個有用的程式設計助手。"),
        ChatMessage(role="user", content="請解釋 Python 中的裝飾器（decorator）是什麼，並提供一個實用的例子。")
    ]
    
    print("正在生成回應...")
    print("回應內容：", end="", flush=True)
    
    response_stream = client.chat(
        model="llama3.2",
        messages=messages,
        stream=True
    )
    
    full_response = ""
    for chunk in response_stream:
        if chunk.get("done", False):
            print(f"\n\n=== 統計資訊 ===")
            print(f"總時間：{chunk.get('total_duration', 0) / 1e9:.2f}秒")
            print(f"載入時間：{chunk.get('load_duration', 0) / 1e9:.2f}秒")
            print(f"生成時間：{chunk.get('eval_duration', 0) / 1e9:.2f}秒")
            print(f"生成字元數：{chunk.get('eval_count', 0)}")
            break
        else:
            message = chunk.get("message", {})
            response_text = message.get("content", "")
            print(response_text, end="", flush=True)
            full_response += response_text
    
    print(f"\n完整回應長度：{len(full_response)} 字元")


def interactive_chat_example():
    """互動式聊天範例"""
    print("\n=== 互動式聊天範例 ===")
    print("輸入 'quit' 或 'exit' 結束對話")
    
    client = OllamaClient()
    
    # 初始化對話記錄
    conversation_history = [
        ChatMessage(role="system", content="你是一個友善的聊天助手。請保持回答簡潔。")
    ]
    
    while True:
        user_input = input("\n你：")
        if user_input.lower() in ['quit', 'exit', '退出']:
            print("再見！")
            break
        
        # 添加用戶訊息到對話記錄
        conversation_history.append(ChatMessage(role="user", content=user_input))
        
        print("助手：", end="", flush=True)
        
        # 生成回應
        response_stream = client.chat(
            model="llama3.2",
            messages=conversation_history,
            stream=True
        )
        
        assistant_response = ""
        for chunk in response_stream:
            if chunk.get("done", False):
                break
            else:
                message = chunk.get("message", {})
                response_text = message.get("content", "")
                print(response_text, end="", flush=True)
                assistant_response += response_text
        
        # 添加助手回應到對話記錄
        conversation_history.append(ChatMessage(role="assistant", content=assistant_response))
        
        print()  # 換行


def streaming_with_progress():
    """帶進度顯示的串流範例"""
    print("\n=== 帶進度顯示的串流範例 ===")
    
    client = OllamaClient()
    
    print("正在生成回應...")
    
    response_stream = client.generate(
        model="llama3.2",
        prompt="寫一篇關於可持續發展的短文，約200字。",
        stream=True
    )
    
    start_time = time.time()
    char_count = 0
    
    print("進度：", end="", flush=True)
    
    for chunk in response_stream:
        if chunk.get("done", False):
            elapsed_time = time.time() - start_time
            print(f"\n\n=== 完成 ===")
            print(f"總字元數：{char_count}")
            print(f"總時間：{elapsed_time:.2f}秒")
            print(f"平均速度：{char_count / elapsed_time:.2f} 字元/秒")
            break
        else:
            response_text = chunk.get("response", "")
            char_count += len(response_text)
            
            # 顯示進度點
            if char_count % 10 == 0:
                print(".", end="", flush=True)
    
    print("\n生成完成！")


if __name__ == "__main__":
    generate_streaming_example()
    chat_streaming_example()
    streaming_with_progress()
    
    # 互動式聊天（註解掉以避免在自動化測試中等待輸入）
    # interactive_chat_example() 