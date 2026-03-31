"""LLM text encoder for Eq. (3-2) and Eq. (3-3)."""

from __future__ import annotations

from types import SimpleNamespace

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = SimpleNamespace(Module=object)

try:
    from peft import LoraConfig, get_peft_model
except ImportError:  # pragma: no cover - optional dependency
    LoraConfig = None
    get_peft_model = None

try:
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
except ImportError:  # pragma: no cover - optional dependency
    AutoModel = None
    AutoTokenizer = None
    BitsAndBytesConfig = None

from lmca_tic.config.schemas import ModelConfig
from lmca_tic.utils.deps import require_dependency


_BaseModule = nn.Module if hasattr(nn, "Module") else object


class HashTextEncoder(_BaseModule):
    def __init__(self, output_dim: int, vocab_size: int = 8192) -> None:
        require_dependency(torch, "torch")
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, output_dim)

    def forward(self, texts: list[str]):
        token_ids = []
        for text in texts:
            hashed = [hash(token) % self.vocab_size for token in text.split()]
            hashed = hashed[:32] or [0]
            token_ids.append(hashed)
        max_len = max(len(ids) for ids in token_ids)
        padded = [ids + [0] * (max_len - len(ids)) for ids in token_ids]
        tensor = torch.tensor(padded, dtype=torch.long, device=self.embedding.weight.device)
        embedded = self.embedding(tensor)
        return embedded.mean(dim=1)


class LLMTextEncoder(_BaseModule):
    def __init__(self, config: ModelConfig, smoke_mode: bool = False) -> None:
        require_dependency(torch, "torch")
        super().__init__()
        self.config = config
        self.smoke_mode = smoke_mode
        self.output_dim = config.embedding_dim
        if smoke_mode or AutoModel is None or AutoTokenizer is None:
            self.backend = HashTextEncoder(config.embedding_dim)
            self.tokenizer = None
            return

        quantization_config = None
        if config.use_4bit and BitsAndBytesConfig is not None:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        model_name = config.smoke_llm_name if smoke_mode else config.llm_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        base_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            quantization_config=quantization_config,
        )
        if (
            not smoke_mode
            and get_peft_model is not None
            and LoraConfig is not None
        ):
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=["q_proj", "v_proj"],
                task_type="FEATURE_EXTRACTION",
            )
            base_model = get_peft_model(base_model, lora_config)
        self.backend = base_model
        hidden_size = int(self.backend.config.hidden_size)
        self.proj = nn.Linear(hidden_size, config.embedding_dim)

    def tokenize_texts(self, texts: list[str], device: torch.device | None = None):
        if texts == []:
            return []
        if self.tokenizer is None:
            return texts
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        if device is not None:
            encoded = {key: value.to(device) for key, value in encoded.items()}
        return encoded

    def forward(self, texts):
        if isinstance(self.backend, HashTextEncoder):
            return self.backend(texts)
        if texts == []:
            return torch.zeros((0, self.output_dim), dtype=self.proj.weight.dtype, device=self.proj.weight.device)
        encoded = texts if isinstance(texts, dict) else self.tokenize_texts(texts)
        device = self.proj.weight.device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = self.backend(**encoded)
        pooled = outputs.last_hidden_state[:, -1, :]
        return self.proj(pooled)
