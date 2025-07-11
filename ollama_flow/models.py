"""
Ollama API 的資料模型定義。
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Generate API 請求模型"""
    model: str = Field(..., description="模型名稱")
    prompt: str = Field(..., description="提示文本")
    suffix: Optional[str] = Field(None, description="模型回應後的文本")
    images: Optional[List[str]] = Field(None, description="Base64 編碼的圖像列表")
    think: Optional[bool] = Field(False, description="是否使用思考模式")
    format: Optional[Union[str, Dict[str, Any]]] = Field(None, description="回應格式（json 或 JSON schema）")
    options: Optional[Dict[str, Any]] = Field(None, description="模型參數")
    system: Optional[str] = Field(None, description="系統訊息")
    template: Optional[str] = Field(None, description="提示模板")
    stream: Optional[bool] = Field(True, description="是否使用串流模式")
    raw: Optional[bool] = Field(False, description="是否使用原始模式")
    keep_alive: Optional[str] = Field("5m", description="模型保持載入的時間")
    context: Optional[List[int]] = Field(None, description="上下文（已棄用）")


class ChatMessage(BaseModel):
    """聊天訊息模型"""
    role: str = Field(..., description="角色：system, user, assistant, tool")
    content: str = Field(..., description="訊息內容")
    images: Optional[List[str]] = Field(None, description="Base64 編碼的圖像列表")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="工具呼叫")
    tool_name: Optional[str] = Field(None, description="工具名稱")


class ChatRequest(BaseModel):
    """Chat API 請求模型"""
    model: str = Field(..., description="模型名稱")
    messages: List[ChatMessage] = Field(..., description="對話訊息")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="工具定義")
    format: Optional[Union[str, Dict[str, Any]]] = Field(None, description="回應格式（json 或 JSON schema）")
    options: Optional[Dict[str, Any]] = Field(None, description="模型參數")
    stream: Optional[bool] = Field(True, description="是否使用串流模式")
    keep_alive: Optional[str] = Field("5m", description="模型保持載入的時間")


class EmbedRequest(BaseModel):
    """Embed API 請求模型"""
    model: str = Field(..., description="模型名稱")
    input: Union[str, List[str]] = Field(..., description="輸入文本或文本列表")
    truncate: Optional[bool] = Field(True, description="是否截斷超出上下文長度的文本")
    options: Optional[Dict[str, Any]] = Field(None, description="模型參數")
    keep_alive: Optional[str] = Field("5m", description="模型保持載入的時間")


class GenerateResponse(BaseModel):
    """Generate API 響應模型"""
    model: str
    created_at: str
    response: str
    done: bool
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None
    done_reason: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat API 響應模型"""
    model: str
    created_at: str
    message: ChatMessage
    done: bool
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None
    done_reason: Optional[str] = None


class EmbedResponse(BaseModel):
    """Embed API 響應模型"""
    model: str
    embeddings: List[List[float]]
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None 