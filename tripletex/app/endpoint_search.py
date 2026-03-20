"""Azure AI Search client for Tripletex endpoint discovery."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


def build_endpoint_index_definition(index_name: str, semantic_configuration: str | None) -> dict[str, Any]:
    definition: dict[str, Any] = {
        "name": index_name,
        "fields": [
            {
                "name": "id",
                "type": "Edm.String",
                "key": True,
                "filterable": True,
                "sortable": True,
                "retrievable": True,
            },
            {
                "name": "path",
                "type": "Edm.String",
                "searchable": True,
                "filterable": True,
                "sortable": True,
                "retrievable": True,
            },
            {
                "name": "method",
                "type": "Edm.String",
                "searchable": True,
                "filterable": True,
                "sortable": True,
                "retrievable": True,
            },
            {
                "name": "operationName",
                "type": "Edm.String",
                "searchable": True,
                "sortable": True,
                "retrievable": True,
            },
            {
                "name": "summary",
                "type": "Edm.String",
                "searchable": True,
                "retrievable": True,
            },
            {
                "name": "description",
                "type": "Edm.String",
                "searchable": True,
                "retrievable": True,
            },
            {
                "name": "tags",
                "type": "Collection(Edm.String)",
                "searchable": True,
                "filterable": True,
                "facetable": True,
                "retrievable": True,
            },
            {
                "name": "content",
                "type": "Edm.String",
                "searchable": True,
                "retrievable": True,
            },
        ],
    }

    if semantic_configuration:
        definition["semantic"] = {
            "configurations": [
                {
                    "name": semantic_configuration,
                    "prioritizedFields": {
                        "titleField": {"fieldName": "operationName"},
                        "prioritizedContentFields": [
                            {"fieldName": "summary"},
                            {"fieldName": "description"},
                            {"fieldName": "content"},
                        ],
                        "prioritizedKeywordsFields": [{"fieldName": "tags"}],
                    },
                }
            ]
        }

    return definition


class EndpointSearchClient:
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        index_name: str,
        api_version: str,
        semantic_configuration: str | None,
        default_top: int,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.index_name = index_name
        self.api_version = api_version
        self.semantic_configuration = semantic_configuration.strip() if semantic_configuration else None
        self.default_top = default_top
        self._client = httpx.AsyncClient(
            base_url=self.endpoint,
            headers={
                "api-key": api_key,
                "Content-Type": "application/json",
            },
            timeout=15.0,
        )

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> EndpointSearchClient | None:
        current = settings or get_settings()
        if not current.azure_search_configured:
            return None

        return cls(
            endpoint=current.azure_search_endpoint or "",
            api_key=current.azure_search_api_key or "",
            index_name=current.azure_search_index_name,
            api_version=current.azure_search_api_version,
            semantic_configuration=current.azure_search_semantic_configuration,
            default_top=current.endpoint_search_results,
        )

    async def search_endpoints(
        self,
        task: str,
        method: str | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        top = min(max(top_k or self.default_top, 1), 10)
        payload = self._build_payload(task=task, method=method, top=top, semantic=True)
        response = await self._client.post(
            f"/indexes('{self.index_name}')/docs/search.post.search",
            params={"api-version": self.api_version},
            json=payload,
        )

        if response.status_code == 400 and self.semantic_configuration:
            logger.info("Semantic search request failed, retrying with simple search")
            payload = self._build_payload(task=task, method=method, top=top, semantic=False)
            response = await self._client.post(
                f"/indexes('{self.index_name}')/docs/search.post.search",
                params={"api-version": self.api_version},
                json=payload,
            )

        if response.status_code == 404:
            raise RuntimeError(
                f"Azure AI Search index '{self.index_name}' was not found. Run the endpoint indexing script first."
            )

        response.raise_for_status()
        body = response.json()

        return {
            "query": task,
            "method": method.upper() if method else None,
            "answers": [answer.get("text") for answer in body.get("@search.answers", []) if answer.get("text")],
            "matches": [self._normalize_match(match) for match in body.get("value", [])],
        }

    async def close(self) -> None:
        await self._client.aclose()

    def _build_payload(
        self,
        task: str,
        method: str | None,
        top: int,
        semantic: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "search": task,
            "top": top,
            "select": "id,path,method,operationName,summary,description,tags,content",
        }

        if method:
            payload["filter"] = f"method eq '{method.upper()}'"

        if semantic and self.semantic_configuration:
            payload.update(
                {
                    "queryType": "semantic",
                    "semanticConfiguration": self.semantic_configuration,
                    "captions": "extractive",
                    "answers": "extractive|count-3",
                }
            )

        return payload

    def _normalize_match(self, match: dict[str, Any]) -> dict[str, Any]:
        captions = [
            item.get("text")
            for item in match.get("@search.captions", [])
            if isinstance(item, dict) and item.get("text")
        ]

        return {
            "id": match.get("id"),
            "path": match.get("path"),
            "method": match.get("method"),
            "operationName": match.get("operationName"),
            "summary": match.get("summary"),
            "description": match.get("description"),
            "tags": match.get("tags") or [],
            "captions": captions,
            "score": match.get("@search.rerankerScore", match.get("@search.score")),
        }