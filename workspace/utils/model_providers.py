CHAT_MODELS = {
    "gpt-economy": {
        "model_id": "gpt-4o-mini",
        "provider": "openai"
    },
    "gpt-latest": {
        "model_id": "gpt-5-mini",
        "provider": "openai"
    },
    "qwen3-compact": {
        "model_id": "qwen3-h2o2-14b",
        "provider": "ollama"
    },
    "qwen3-mid": {
        "model_id": "qwen3-h2o2-30b",
        "provider": "ollama"
    }
}
EMBEDDING_MODELS = {
    "gpt-emb": {
        "model_id": "text-embedding-3-small",
        "provider": "openai"
    },
    "qwen-emb": {
        "model_id": "openhermes",
        "provider": "ollama"
    }        
}