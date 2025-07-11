"""
Ollama API 客戶端實現。
"""

import json
import requests
from typing import Union, Dict, Any, Iterator, Optional, Type, List
from urllib.parse import urljoin

from .models import (
    GenerateRequest, GenerateResponse,
    ChatRequest, ChatResponse, ChatMessage,
    EmbedRequest, EmbedResponse
)
from .schemas import StructuredOutput
from pydantic import BaseModel


class OllamaClient:
    """
    Ollama API 客戶端類別。
    
    支援功能：
    - Generate API（生成完成）
    - Chat API（聊天完成）
    - Embed API（生成嵌入）
    - 結構化輸出
    - 串流模式
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 30, check_models: bool = True):
        """
        初始化 Ollama 客戶端。
        
        Args:
            base_url: Ollama 服務器的基礎 URL
            timeout: 請求超時時間（秒）
            check_models: 是否在調用前檢查模型是否存在
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.check_models = check_models
        self.session = requests.Session()
        self._models_cache = None  # 模型列表緩存
        
        # 設定請求頭
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def _make_request(self, endpoint: str, data: Optional[Dict[str, Any]] = None, stream: bool = False, method: str = "POST") -> requests.Response:
        """
        發送 HTTP 請求。
        
        Args:
            endpoint: API 端點
            data: 請求資料（可選）
            stream: 是否使用串流模式
            method: HTTP 方法（GET 或 POST）
            
        Returns:
            HTTP 響應對象
        """
        url = urljoin(self.base_url, endpoint)
        
        try:
            if method.upper() == "GET":
                response = self.session.get(
                    url,
                    timeout=self.timeout,
                    stream=stream
                )
            else:
                response = self.session.post(
                    url,
                    json=data,
                    timeout=self.timeout,
                    stream=stream
                )
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"請求失敗：{e}")
    
    def _stream_response(self, response: requests.Response) -> Iterator[Dict[str, Any]]:
        """
        處理串流響應。
        
        Args:
            response: HTTP 響應對象
            
        Yields:
            每個 JSON 響應對象
        """
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    
    def list_models(self, refresh_cache: bool = False) -> List[str]:
        """
        獲取可用模型列表。
        
        Args:
            refresh_cache: 是否刷新緩存
            
        Returns:
            模型名稱列表
        """
        if self._models_cache is None or refresh_cache:
            try:
                response = self._make_request("/api/tags", method="GET")
                data = response.json()
                
                models = []
                for model_info in data.get("models", []):
                    models.append(model_info.get("name", ""))
                
                self._models_cache = models
                return models
            except Exception as e:
                raise Exception(f"獲取模型列表失敗：{e}")
        
        return self._models_cache or []
    
    def _check_model_exists(self, model: str) -> None:
        """
        檢查模型是否存在。
        
        Args:
            model: 模型名稱
            
        Raises:
            ValueError: 如果模型不存在
        """
        if not self.check_models:
            return
        
        available_models = self.list_models()
        if model not in available_models:
            raise ValueError(
                f"模型 '{model}' 不存在。可用模型：{', '.join(available_models)}"
            )
    
    def refresh_models_cache(self) -> List[str]:
        """
        刷新模型緩存並返回最新的模型列表。
        
        Returns:
            最新的模型名稱列表
        """
        return self.list_models(refresh_cache=True)
    
    def generate(
        self,
        model: str,
        prompt: str,
        *,
        suffix: Optional[str] = None,
        images: Optional[List[str]] = None,
        think: bool = False,
        format: Optional[Union[str, Dict[str, Any]]] = None,
        options: Optional[Dict[str, Any]] = None,
        system: Optional[str] = None,
        template: Optional[str] = None,
        stream: bool = False,
        raw: bool = False,
        keep_alive: str = "5m",
        context: Optional[List[int]] = None
    ) -> Union[GenerateResponse, Iterator[Dict[str, Any]]]:
        """
        生成完成（Generate API）。
        
        Args:
            model: 模型名稱
            prompt: 提示文本
            suffix: 模型回應後的文本
            images: Base64 編碼的圖像列表
            think: 是否使用思考模式
            format: 回應格式（json 或 JSON schema）
            options: 模型參數
            system: 系統訊息
            template: 提示模板
            stream: 是否使用串流模式
            raw: 是否使用原始模式
            keep_alive: 模型保持載入的時間
            context: 上下文（已棄用）
            
        Returns:
            GenerateResponse 或串流響應迭代器
        """
        # 檢查模型是否存在
        self._check_model_exists(model)
        
        request = GenerateRequest(
            model=model,
            prompt=prompt,
            suffix=suffix,
            images=images,
            think=think,
            format=format,
            options=options,
            system=system,
            template=template,
            stream=stream,
            raw=raw,
            keep_alive=keep_alive,
            context=context
        )
        
        response = self._make_request("/api/generate", request.model_dump(exclude_none=True), stream=stream)
        
        if stream:
            return self._stream_response(response)
        else:
            data = response.json()
            return GenerateResponse.model_validate(data)
    
    def chat(
        self,
        model: str,
        messages: List[Union[ChatMessage, Dict[str, Any]]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        format: Optional[Union[str, Dict[str, Any]]] = None,
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        keep_alive: str = "5m"
    ) -> Union[ChatResponse, Iterator[Dict[str, Any]]]:
        """
        聊天完成（Chat API）。
        
        Args:
            model: 模型名稱
            messages: 對話訊息列表
            tools: 工具定義
            format: 回應格式（json 或 JSON schema）
            options: 模型參數
            stream: 是否使用串流模式
            keep_alive: 模型保持載入的時間
            
        Returns:
            ChatResponse 或串流響應迭代器
        """
        # 檢查模型是否存在
        self._check_model_exists(model)
        
        # 轉換訊息格式
        chat_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                chat_messages.append(ChatMessage.model_validate(msg))
            else:
                chat_messages.append(msg)
        
        request = ChatRequest(
            model=model,
            messages=chat_messages,
            tools=tools,
            format=format,
            options=options,
            stream=stream,
            keep_alive=keep_alive
        )
        
        response = self._make_request("/api/chat", request.model_dump(exclude_none=True), stream=stream)
        
        if stream:
            return self._stream_response(response)
        else:
            data = response.json()
            return ChatResponse.model_validate(data)
    
    def embed(
        self,
        model: str,
        input: Union[str, List[str]],
        *,
        truncate: bool = True,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: str = "5m"
    ) -> EmbedResponse:
        """
        生成嵌入（Embed API）。
        
        Args:
            model: 模型名稱
            input: 輸入文本或文本列表
            truncate: 是否截斷超出上下文長度的文本
            options: 模型參數
            keep_alive: 模型保持載入的時間
            
        Returns:
            EmbedResponse
        """
        # 檢查模型是否存在
        self._check_model_exists(model)
        
        request = EmbedRequest(
            model=model,
            input=input,
            truncate=truncate,
            options=options,
            keep_alive=keep_alive
        )
        
        response = self._make_request("/api/embed", request.model_dump(exclude_none=True))
        data = response.json()
        return EmbedResponse.model_validate(data)
    
    # 便利方法
    def generate_json(
        self,
        model: str,
        prompt: str,
        **kwargs
    ) -> Union[GenerateResponse, Iterator[Dict[str, Any]]]:
        """
        生成 JSON 格式的回應。
        
        Args:
            model: 模型名稱
            prompt: 提示文本
            **kwargs: 其他參數
            
        Returns:
            GenerateResponse 或串流響應迭代器
        """
        return self.generate(model, prompt, format="json", **kwargs)
    
    def generate_structured(
        self,
        model: str,
        prompt: str,
        schema: Union[Type[BaseModel], Dict[str, Any]],
        **kwargs
    ) -> Union[GenerateResponse, Iterator[Dict[str, Any]]]:
        """
        生成結構化回應。
        
        Args:
            model: 模型名稱
            prompt: 提示文本
            schema: Pydantic 模型類別或 JSON schema 字典
            **kwargs: 其他參數
            
        Returns:
            GenerateResponse 或串流響應迭代器
        """
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            format_schema = StructuredOutput.from_pydantic(schema)
        else:
            format_schema = schema
            
        return self.generate(model, prompt, format=format_schema, **kwargs)
    
    def chat_json(
        self,
        model: str,
        messages: List[Union[ChatMessage, Dict[str, Any]]],
        **kwargs
    ) -> Union[ChatResponse, Iterator[Dict[str, Any]]]:
        """
        聊天生成 JSON 格式的回應。
        
        Args:
            model: 模型名稱
            messages: 對話訊息列表
            **kwargs: 其他參數
            
        Returns:
            ChatResponse 或串流響應迭代器
        """
        return self.chat(model, messages, format="json", **kwargs)
    
    def chat_structured(
        self,
        model: str,
        messages: List[Union[ChatMessage, Dict[str, Any]]],
        schema: Union[Type[BaseModel], Dict[str, Any]],
        **kwargs
    ) -> Union[ChatResponse, Iterator[Dict[str, Any]]]:
        """
        聊天生成結構化回應。
        
        Args:
            model: 模型名稱
            messages: 對話訊息列表
            schema: Pydantic 模型類別或 JSON schema 字典
            **kwargs: 其他參數
            
        Returns:
            ChatResponse 或串流響應迭代器
        """
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            format_schema = StructuredOutput.from_pydantic(schema)
        else:
            format_schema = schema
            
        return self.chat(model, messages, format=format_schema, **kwargs)
    
    def parse_structured_response(
        self,
        response: str,
        model_class: Optional[Type[BaseModel]] = None
    ) -> Any:
        """
        解析結構化響應。
        
        Args:
            response: 響應字符串
            model_class: 可選的 Pydantic 模型類別
            
        Returns:
            解析後的對象
        """
        return StructuredOutput.parse_response(response, model_class)
    
    def __enter__(self):
        """上下文管理器進入"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.session.close() 