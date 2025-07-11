"""
Ollama Flow 使用範例
"""

from ollama_flow import OllamaClient, ChatMessage, StructuredOutput
from pydantic import BaseModel, Field
from typing import List


class PersonInfo(BaseModel):
    """人物資訊模型"""
    name: str = Field(..., description="姓名")
    age: int = Field(..., description="年齡")
    occupation: str = Field(..., description="職業")
    skills: List[str] = Field(..., description="技能列表")


def main():
    print("=== Ollama Flow 使用範例 ===")
    
    # 建立客戶端
    client = OllamaClient(base_url="http://localhost:11434")
    
    try:
        # 範例 1: 基本生成
        print("\n1. 基本生成範例：")
        response = client.generate(
            model="qwen3:4b-q4_K_M",
            prompt="解釋什麼是機器學習？",
            stream=False
        )
        print(f"回應：{response.response}")
        
        # 範例 2: 聊天對話
        print("\n2. 聊天對話範例：")
        messages = [
            ChatMessage(role="user", content="你好！你是誰？")
        ]
        chat_response = client.chat(
            model="qwen3:4b-q4_K_M",
            messages=messages,
            stream=False
        )
        print(f"回應：{chat_response.message.content}")
        
        # 範例 3: 結構化輸出
        print("\n3. 結構化輸出範例：")
        structured_response = client.generate_structured(
            model="qwen3:4b-q4_K_M",
            prompt="介紹一個虛構的軟體工程師角色。請以 JSON 格式回應。",
            schema=PersonInfo,
            stream=False
        )
        print(f"結構化回應：{structured_response.response}")
        
        # 解析結構化回應
        person_data = client.parse_structured_response(
            structured_response.response,
            PersonInfo
        )
        print(f"解析後的資料：{person_data}")
        
        # 範例 4: JSON 模式
        print("\n4. JSON 模式範例：")
        json_response = client.generate_json(
            model="qwen3:4b-q4_K_M",
            prompt="列出三個程式設計語言及其特點。請以 JSON 格式回應。",
            stream=False
        )
        print(f"JSON 回應：{json_response.response}")
        
        # 範例 5: 生成嵌入
        print("\n5. 生成嵌入範例：")
        embed_response = client.embed(
            model="bge-m3:latest",
            input="這是一段測試文本"
        )
        print(f"嵌入維度：{len(embed_response.embeddings[0])}")
        
    except Exception as e:
        print(f"錯誤：{e}")
        print("請確保 Ollama 服務正在運行，並且已安裝所需的模型。")


if __name__ == "__main__":
    main()
