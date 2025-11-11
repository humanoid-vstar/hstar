# vagen/mllm_agent/model_interface/openai/model_config.py
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from vagen.inference.model_interface.base_model_config import BaseModelConfig

@dataclass
class OpenAIModelConfig(BaseModelConfig):
    """Configuration for OpenAI API model interface."""
    
    # OpenAI specific parameters
    api_key: Optional[str] = None  # If None, will use environment variable
    organization: Optional[str] = None
    base_url: Optional[str] = None  # For custom endpoints
    window_size = 5
    # Model parameters
    model_name: str = "gpt-4o"
    max_retries: int = 3
    timeout: int = 60
    
    # Generation parameters (inherited from base)
    # max_tokens, temperature already defined in base
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    
    # Provider identifier
    provider: str = "openai"
    
    def config_id(self) -> str:
        """Generate unique identifier for this configuration."""
        return f"OpenAIModelConfig({self.model_name},max_tokens={self.max_tokens},temp={self.temperature})"
    
    @staticmethod
    def get_provider_info() -> Dict[str, Any]:
        """Get information about the OpenAI provider."""
        return {
            "description": "OpenAI API for GPT models",
            "supports_multimodal": True,
            "supported_models": [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo", 
                "gpt-4-vision-preview",
                "gpt-3.5-turbo",
                "Qwen/Qwen2.5-VL-3B-Instruct",
                "Qwen/Qwen2.5-VL-7B-Instruct",
                "OpenGVLab/InternVL3_5-8B",
                'OpenGVLab/InternVL3_5-4B',
                "google/gemma-3-4b-it",
                "deepseek-ai/deepseek-vl2-tiny"
            ],
            "default_model": "gpt-4o"
        }