import os
import json
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from config import config

class BaseLLMProvider(ABC):
    """Abstract interface for local Large Language Model providers."""
    
    @abstractmethod
    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        pass
        
    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

class FoundryLocalProvider(BaseLLMProvider):
    """Provider for Microsoft Foundry Local SDK (on-device CPU/NPU model inference)."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.foundry_model_name
        self._client = None
        self._model = None
        self._init_sdk()

    def _init_sdk(self):
        try:
            from foundry_local import FoundryClient
            self._client = FoundryClient()
            self._model = self._client.load_model(self.model_name)
        except Exception as e:
            raise RuntimeError(f"Microsoft Foundry Local SDK unavailable: {e}")

    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        if not self._model:
            raise RuntimeError("Foundry Local model is not loaded.")
        
        full_prompt = f"System: {system_prompt}\n\nUser Question & Context:\n{user_prompt}"
        response = self._model.complete(full_prompt)
        return str(response)

    @property
    def provider_name(self) -> str:
        return f"Microsoft Foundry Local ({self.model_name})"

class LocalOpenAIProvider(BaseLLMProvider):
    """Provider for local OpenAI-compatible HTTP servers (e.g. Ollama, LM Studio, vLLM)."""
    
    def __init__(self, base_url: str = None, api_key: str = None):
        self.base_url = (base_url or config.openai_base_url).rstrip('/')
        self.api_key = api_key or config.api_key if hasattr(config, 'api_key') else 'ollama'
        self._ping()

    def _ping(self):
        url = f"{self.base_url}/models"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                pass
        except Exception as e:
            raise RuntimeError(f"Local OpenAI endpoint unavailable at {url}: {e}")

    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": "phi-3.5-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers=headers,
            method='POST'
        )
        
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                return result['choices'][0]['message']['content']
        except Exception as e:
            raise RuntimeError(f"Local OpenAI endpoint error at {url}: {e}")

    @property
    def provider_name(self) -> str:
        return f"Local OpenAI API ({self.base_url})"

class OfflineFallbackProvider(BaseLLMProvider):
    """Self-contained lightweight offline generation engine for instant execution without hardware setup."""
    
    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        if "Context Passages:" not in user_prompt or "No relevant document passages found" in user_prompt:
            return (
                "I do not have enough information in the local knowledge base to answer this question.\n"
                "(Yerel bilgi tabanında bu soruyla ilgili yeterli bilgi bulunmamaktadır.)\n\n"
                "*Note: Please ensure relevant documents are ingested into the database.*"
            )

        try:
            context_part = user_prompt.split("Context Passages:")[1].split("Question:")[0].strip()
            passages = [p.strip() for p in context_part.split("\n\n---\n\n") if p.strip()]
        except Exception:
            passages = [user_prompt]

        if not passages:
            return (
                "I do not have enough information in the local knowledge base to answer this question.\n"
                "(Yerel bilgi tabanında bu soruyla ilgili yeterli bilgi bulunmamaktadır.)"
            )

        lines = ["Based on the local knowledge base:"]
        for p in passages:
            lines.append(f"\n{p}")
            
        return "\n".join(lines)

    @property
    def provider_name(self) -> str:
        return "Offline Grounded Engine (Standard Local Fallback)"

def get_llm_provider(mode: str = None) -> BaseLLMProvider:
    """Factory function for selecting active LLM provider with fallback handling."""
    mode = mode or config.llm_provider
    
    if mode == "foundry":
        try:
            return FoundryLocalProvider()
        except Exception as e:
            print(f"[Warning] Foundry Local SDK not ready: {e}. Falling back to Offline Provider.")
            return OfflineFallbackProvider()
            
    elif mode == "openai":
        try:
            return LocalOpenAIProvider()
        except Exception as e:
            print(f"[Warning] Local OpenAI server unavailable: {e}. Falling back to Offline Provider.")
            return OfflineFallbackProvider()

    elif mode == "auto":
        try:
            return FoundryLocalProvider()
        except Exception:
            pass
            
        try:
            return LocalOpenAIProvider()
        except Exception:
            pass

    return OfflineFallbackProvider()
