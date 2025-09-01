from pydantic_settings import BaseSettings


class TeamSettings(BaseSettings):
    """Team settings that can be set using environment variables.

    Reference: https://pydantic-docs.helpmanual.io/usage/settings/
    """

    gpt_4_mini: str = "gpt-4o-mini"
    gpt_4: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"

    qwen: str = "qwen3:30b"

    default_max_completion_tokens: int = 16000
    default_temperature: float = 0


# Create an TeamSettings object
team_settings = TeamSettings()
