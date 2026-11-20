from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

# from langchain_core.language_models.fake_chat_models import FakeListChatModel
from bookworm.config import Settings
from bookworm.config import settings as default_settings


def create_llm(settings: Settings | None = None) -> BaseChatModel:

    if settings is None:
        settings = default_settings

    if settings.llm_provider == "ollama":
        return ChatOllama(
            model       = settings.llm_model,
            base_url    = settings.ollama_base_url,
            temperature = settings.llm_temperature,
            num_predict = settings.llm_max_tokens,
            validate_model_on_init = True,
        )

    # elif settings.llm_provider == "mock":
    #     return FakeListChatModel(responses = ["Hello! I'm Bookworm, your library assistant."])

    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
