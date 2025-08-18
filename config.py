"""
config.py - Configuration classes for LLM and retry settings
"""

from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Configuration for LLM settings"""
    ##### for vllm
    # model_dir: str = '/root/.cache/kagglehub/models/metaresearch/llama-3.2/transformers/3b-instruct/1'
    ##### for ollama
    model_name: str = "gpt-oss:20b"
    host: str = "http://localhost:11434"
    max_concurrency: int = 4
    ##################

    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.80
    max_model_len: int = 8192
    trust_remote_code: bool = True
    dtype: str = "half"
    enforce_eager: bool = True

    # Sampling parameters
    n_samples: int = 1
    temperature: float = 0.8 # 0~2.0 for vllm, 0~1.0 for ollama
    top_p: float = 0.9
    seed: int = 777
    max_tokens: int = 2048


@dataclass
class RetryConfig:
    """Configuration for retry logic"""
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    temperature_increment: float = 0.1  # Increase temperature on retry