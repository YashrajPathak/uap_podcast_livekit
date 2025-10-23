from dotenv import load_dotenv, set_key
import os
import time
import httpx
from typing import Optional
from .logging import default_logger as logger

class TokenManager:
    """
    Manages OAuth2 token generation and caching.
    Tokens are cached in environment variables with expiry handling.
    """
    def __init__(self, auth_url: str, grant_type: str, scope: str, env_path=".env"):
        self.auth_url           = auth_url
        self.grant_type         = grant_type
        self.scope              = scope
        self.token_expiry_key   = "AZURE_OPENAI_TOKEN_EXPIRY"
        self.token_key          = "AZURE_OPENAI_API_KEY"
        self.env_path           = env_path

        load_dotenv(self.env_path)

    def _is_token_valid(self) -> bool:
        token   = os.environ.get(self.token_key)
        expiry  = os.environ.get(self.token_expiry_key)
        if token and expiry:
            try:
                return time.time() < float(expiry)
            except ValueError:
                return False
        return False

    def _update_env(self, key, value):
        set_key(self.env_path, key, value)
        os.environ[key] = value

    async def generate_token(self) -> Optional[str]:
        if self._is_token_valid():
            return os.environ.get(self.token_key)

        body = {
            "grant_type": self.grant_type,
            "scope": self.scope,
            "client_id": os.environ.get("LLM_CLIENT_ID"),
            "client_secret": os.environ.get("LLM_CLIENT_SECRET"),
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(self.auth_url, headers=headers, data=body)
                response.raise_for_status()
                data = response.json()

                access_token = data.get("access_token")
                expires_in = data.get("expires_in", 3600)

                if access_token:
                    expiry_time = str(time.time() + expires_in)
                    self._update_env(self.token_key, access_token)
                    self._update_env(self.token_expiry_key, expiry_time)
                    return access_token
                else:
                    logger.error("No access token found in response.")
                    return None
        except httpx.HTTPError as e:
            logger.error(f"HTTP error occurred: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None
        
if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    load_dotenv()

    async def main():
        auth_url           = os.getenv("AUTH_URL", "https://api.uhg.com/oauth2/token")
        grant_type         = os.getenv("GRANT_TYPE", "client_credentials")
        scope              = os.getenv("SCOPE", "https://api.uhg.com/.default")

        token_manager      = TokenManager(auth_url=auth_url, grant_type=grant_type, scope=scope)
        token              = await token_manager.generate_token()

        if token:
            logger.info(f"Generated Token: {token}")
        else:
            logger.error("Failed to generate token.")

    asyncio.run(main())
