"""
結構化輸出支援模組。
"""

from typing import Dict, Any, Type, Optional
from pydantic import BaseModel
import json


class StructuredOutput:
    """
    結構化輸出輔助類別，用於生成 JSON Schema 並處理結構化響應。
    """
    
    @staticmethod
    def from_pydantic(model_class: Type[BaseModel]) -> Dict[str, Any]:
        """
        從 Pydantic 模型生成 JSON Schema。
        
        Args:
            model_class: Pydantic 模型類別
            
        Returns:
            JSON Schema 字典
        """
        return model_class.model_json_schema()
    
    @staticmethod
    def from_dict(schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        從字典生成 JSON Schema。
        
        Args:
            schema: Schema 字典
            
        Returns:
            JSON Schema 字典
        """
        return schema
    
    @staticmethod
    def json_mode() -> str:
        """
        返回 JSON 模式標識符。
        
        Returns:
            "json" 字符串
        """
        return "json"
    
    @staticmethod
    def parse_response(response: str, model_class: Optional[Type[BaseModel]] = None) -> Any:
        """
        解析結構化響應。
        
        Args:
            response: 響應字符串
            model_class: 可選的 Pydantic 模型類別
            
        Returns:
            解析後的對象
        """
        try:
            data = json.loads(response)
            if model_class:
                return model_class.model_validate(data)
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"無法解析 JSON 響應：{e}")
        except Exception as e:
            raise ValueError(f"無法驗證響應資料：{e}")


# 便利函數
def create_json_schema(model_class: Type[BaseModel]) -> Dict[str, Any]:
    """
    建立 JSON Schema 的便利函數。
    
    Args:
        model_class: Pydantic 模型類別
        
    Returns:
        JSON Schema 字典
    """
    return StructuredOutput.from_pydantic(model_class)


def json_format() -> str:
    """
    返回 JSON 格式標識符的便利函數。
    
    Returns:
        "json" 字符串
    """
    return StructuredOutput.json_mode()


# 常用的 JSON Schema 範例
COMMON_SCHEMAS = {
    "person": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string", "format": "email"}
        },
        "required": ["name", "age"]
    },
    "product": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "price": {"type": "number"},
            "category": {"type": "string"},
            "in_stock": {"type": "boolean"}
        },
        "required": ["name", "price"]
    },
    "summary": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "content": {"type": "string"},
            "keywords": {"type": "array", "items": {"type": "string"}},
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]}
        },
        "required": ["title", "content"]
    }
} 