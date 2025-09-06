from pydantic_settings import BaseSettings
from workspace.utils.model_providers import CHAT_MODELS, EMBEDDING_MODELS

class TeamSettings(BaseSettings):
    """Team settings that can be set using environment variables.

    Reference: https://pydantic-docs.helpmanual.io/usage/settings/
    """

    openai_economy: str = CHAT_MODELS.get('gpt-economy').get("model_id")
    openai_latest: str = CHAT_MODELS.get('gpt-latest').get("model_id")
    openai_embedding_model: str = EMBEDDING_MODELS.get('gpt-emb').get("model_id")

    ollama_compact: str = CHAT_MODELS.get('qwen3-compact').get("model_id")
    ollama_mid: str = CHAT_MODELS.get('qwen3-mid').get("model_id")
    ollama_embedding_model: str = EMBEDDING_MODELS.get('qwen-emb').get("model_id")

    default_max_completion_tokens: int = 16000
    default_temperature: float = 0


# Create an TeamSettings object
team_settings = TeamSettings()
