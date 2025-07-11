"""
結構化輸出範例
"""

from ollama_flow import OllamaClient, ChatMessage, StructuredOutput
from pydantic import BaseModel, Field
from typing import List, Optional
import json


class Product(BaseModel):
    """產品資訊模型"""
    name: str = Field(..., description="產品名稱")
    price: float = Field(..., description="價格")
    category: str = Field(..., description="分類")
    description: str = Field(..., description="描述")
    features: List[str] = Field(..., description="功能特點")
    in_stock: bool = Field(..., description="是否有庫存")


class BookSummary(BaseModel):
    """書籍摘要模型"""
    title: str = Field(..., description="書名")
    author: str = Field(..., description="作者")
    genre: str = Field(..., description="類型")
    main_themes: List[str] = Field(..., description="主要主題")
    rating: int = Field(..., description="評分 (1-10)")
    summary: str = Field(..., description="摘要")


class WeatherInfo(BaseModel):
    """天氣資訊模型"""
    location: str = Field(..., description="地點")
    temperature: float = Field(..., description="溫度")
    humidity: int = Field(..., description="濕度百分比")
    condition: str = Field(..., description="天氣狀況")
    wind_speed: Optional[float] = Field(None, description="風速")


def pydantic_schema_example():
    """使用 Pydantic 模型的結構化輸出範例"""
    print("=== Pydantic 模型結構化輸出範例 ===")
    
    client = OllamaClient()
    
    # 生成產品資訊
    response = client.generate_structured(
        model="llama3.2",
        prompt="創建一個智慧型手機的產品資訊。請用 JSON 格式回應。",
        schema=Product,
        stream=False
    )
    
    print(f"原始回應：{response.response}")
    
    # 解析結構化回應
    try:
        product = client.parse_structured_response(response.response, Product)
        print(f"\n解析後的產品資訊：")
        print(f"名稱：{product.name}")
        print(f"價格：${product.price}")
        print(f"分類：{product.category}")
        print(f"描述：{product.description}")
        print(f"功能：{', '.join(product.features)}")
        print(f"庫存：{'有' if product.in_stock else '無'}")
    except Exception as e:
        print(f"解析錯誤：{e}")


def json_schema_example():
    """使用 JSON Schema 字典的結構化輸出範例"""
    print("\n=== JSON Schema 結構化輸出範例 ===")
    
    client = OllamaClient()
    
    # 自定義 JSON Schema
    person_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "occupation": {"type": "string"},
            "hobbies": {"type": "array", "items": {"type": "string"}},
            "married": {"type": "boolean"}
        },
        "required": ["name", "age", "occupation"]
    }
    
    response = client.generate_structured(
        model="llama3.2",
        prompt="創建一個虛構角色的個人資訊。請用 JSON 格式回應。",
        schema=person_schema,
        stream=False
    )
    
    print(f"原始回應：{response.response}")
    
    # 解析回應
    try:
        person_data = json.loads(response.response)
        print(f"\n解析後的角色資訊：")
        print(f"姓名：{person_data.get('name', 'N/A')}")
        print(f"年齡：{person_data.get('age', 'N/A')}")
        print(f"職業：{person_data.get('occupation', 'N/A')}")
        print(f"愛好：{', '.join(person_data.get('hobbies', []))}")
        print(f"已婚：{'是' if person_data.get('married', False) else '否'}")
    except json.JSONDecodeError as e:
        print(f"JSON 解析錯誤：{e}")


def chat_structured_example():
    """聊天模式的結構化輸出範例"""
    print("\n=== 聊天模式結構化輸出範例 ===")
    
    client = OllamaClient()
    
    messages = [
        ChatMessage(role="system", content="你是一個書籍推薦助手。"),
        ChatMessage(role="user", content="推薦一本科幻小說，並提供詳細資訊。請用 JSON 格式回應。")
    ]
    
    response = client.chat_structured(
        model="llama3.2",
        messages=messages,
        schema=BookSummary,
        stream=False
    )
    
    print(f"原始回應：{response.message.content}")
    
    # 解析結構化回應
    try:
        book = client.parse_structured_response(response.message.content, BookSummary)
        print(f"\n推薦書籍：")
        print(f"書名：{book.title}")
        print(f"作者：{book.author}")
        print(f"類型：{book.genre}")
        print(f"主要主題：{', '.join(book.main_themes)}")
        print(f"評分：{book.rating}/10")
        print(f"摘要：{book.summary}")
    except Exception as e:
        print(f"解析錯誤：{e}")


def json_mode_example():
    """JSON 模式範例"""
    print("\n=== JSON 模式範例 ===")
    
    client = OllamaClient()
    
    response = client.generate_json(
        model="llama3.2",
        prompt="列出三個程式設計語言及其主要特點。請用 JSON 格式回應。",
        stream=False
    )
    
    print(f"原始回應：{response.response}")
    
    # 解析 JSON 回應
    try:
        data = json.loads(response.response)
        print(f"\n解析後的資料：")
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except json.JSONDecodeError as e:
        print(f"JSON 解析錯誤：{e}")


if __name__ == "__main__":
    pydantic_schema_example()
    json_schema_example()
    chat_structured_example()
    json_mode_example() 