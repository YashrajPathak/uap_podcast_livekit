# Adapter module to expose snake_case import path
from .Models_podcast import *  # noqa: F401,F403

# Provide a placeholder symbol for tests that patch AzureOpenAI on this module
class AzureOpenAI:  # pragma: no cover
    pass
