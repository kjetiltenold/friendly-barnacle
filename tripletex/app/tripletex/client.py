import logging

import httpx

logger = logging.getLogger(__name__)


class TripletexClient:
    def __init__(self, base_url: str, session_token: str):
        self.base_url = base_url.rstrip("/")
        self._auth = httpx.BasicAuth(username="0", password=session_token)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            auth=self._auth,
            timeout=30.0,
        )
        self.call_count = 0
        self.error_count = 0
        self.write_call_count = 0

    async def get(self, path: str, params: dict | None = None) -> dict:
        return await self._request("GET", path, params=params)

    async def post(self, path: str, json: dict | None = None) -> dict:
        return await self._request("POST", path, json=json)

    async def put(self, path: str, json: dict | None = None, params: dict | None = None) -> dict:
        return await self._request("PUT", path, json=json, params=params)

    async def delete(self, path: str) -> dict:
        return await self._request("DELETE", path)

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        self.call_count += 1
        if method in {"POST", "PUT", "DELETE"}:
            self.write_call_count += 1
        logger.info(f"Tripletex {method} {path} (call #{self.call_count})")
        resp = await self._client.request(method, path, **kwargs)
        if resp.status_code >= 400:
            self.error_count += 1
            body = resp.text[:500]
            logger.warning(f"Tripletex {method} {path} -> {resp.status_code}: {body}")
            # Raise with response body in message so callers can match on it
            raise httpx.HTTPStatusError(
                f"{resp.status_code} {resp.reason_phrase}: {body}",
                request=resp.request,
                response=resp,
            )
        if resp.status_code == 204 or not resp.content:
            return {}
        return resp.json()

    async def close(self):
        await self._client.aclose()
