import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from .token_manager import TokenManager  
from .logging import default_logger as logger

class LLMConfig:
    """
    Loads application configuration from environment variables.
    """
    def __init__(self):
        load_dotenv()
        self.endpoint           = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version        = os.getenv("AZURE_OPENAI_API_VERSION", "2023-06-01-preview")
        self.deployment         = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-35-turbo")
        self.project_id         = os.getenv("PROJECT_ID")
        self.client_id          = os.getenv("LLM_CLIENT_ID")
        self.client_secret      = os.getenv("LLM_CLIENT_SECRET")
        self.idp                = os.getenv("AOG_GATEWAY_IDP", "azuread")
        self.auth_url           = os.getenv("LLM_AUTH_URL", "https://api.uhg.com/oauth2/token")
        self.grant_type         = os.getenv("LLM_GRANT_TYPE", "client_credentials")
        self.scope              = os.getenv("LLM_SCOPE", "https://api.uhg.com/.default")

    def default_headers(self):
        """
        Returns default headers for API requests.
        """
        headers = {}
        if self.project_id:
            headers["projectId"] = self.project_id
        if self.idp:
            headers["x-idp"]     = self.idp
        return headers
    

class LLMFactory:
    """
    Factory for creating AzureChatOpenAI LLM instances with dynamic token management.
    """
    def __init__(self, cfg: LLMConfig, token_mgr: TokenManager):
        self.cfg        = cfg
        self.token_mgr  = token_mgr

    async def create_llm(self):
        """
        Asynchronously creates and returns an AzureChatOpenAI instance.
        """
        token = await self.token_mgr.generate_token()
        if not token:
            logger.error("Failed to obtain access token for Azure OpenAI.")
            raise RuntimeError("Failed to obtain access token for Azure OpenAI.")
        else:
            #logger.info("Successfully obtained access token for Azure OpenAI." + str(token))
            # Set for LangChain usage
            os.environ["AZURE_OPENAI_API_KEY"] = token 

        llm = AzureChatOpenAI(
            azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            default_headers={
                "projectId": os.environ.get("PROJECT_ID")
            }
        )
        return llm

async def main():
    """
    Example usage of LLMFactory to invoke a simple prompt.
    """
    cfg             = LLMConfig()
    token_mgr       = TokenManager(auth_url=cfg.auth_url, grant_type=cfg.grant_type, scope=cfg.scope)
    llm             = await LLMFactory(cfg, token_mgr).create_llm()

    response        = await llm.ainvoke("What is a prime number?")
    logger.info(f"LLM response: {response}")

if __name__ == "__main__":
    asyncio.run(main())
