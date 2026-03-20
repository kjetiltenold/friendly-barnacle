"""Index Tripletex endpoint metadata into Azure AI Search.

Usage:
    python scripts/index_tripletex_endpoints.py [path\to\tripletex_endpoints.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_JSON_FILE = PROJECT_DIR / "openapi.json"

if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_DIR))

import httpx

from app.config import get_settings
from app.endpoint_search import build_endpoint_index_definition

HTTP_METHODS = {"get", "post", "put", "delete", "patch", "head", "options"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json_file",
        nargs="?",
        type=Path,
        default=DEFAULT_JSON_FILE,
        help="Path to the JSON file containing Tripletex endpoints. Defaults to openapi.json in the project root.",
    )
    parser.add_argument("--batch-size", type=int, default=500, help="Documents per upload batch")
    parser.add_argument(
        "--recreate-index",
        action="store_true",
        help="Delete and recreate the search index before uploading documents",
    )
    args = parser.parse_args()
    json_file = args.json_file.resolve()

    settings = get_settings()
    if not settings.azure_search_configured:
        raise SystemExit("AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY must be configured in the environment")
    if not json_file.exists():
        raise SystemExit(f"JSON file not found: {json_file}")

    payload = json.loads(json_file.read_text(encoding="utf-8"))
    documents = extract_endpoint_documents(payload)
    if not documents:
        raise SystemExit("No endpoint documents could be extracted from the JSON file")

    with httpx.Client(
        base_url=(settings.azure_search_endpoint or "").rstrip("/"),
        headers={
            "api-key": settings.azure_search_api_key or "",
            "Content-Type": "application/json",
        },
        timeout=60.0,
    ) as client:
        ensure_index(client, settings, recreate=args.recreate_index)
        upload_documents(client, settings, documents, batch_size=args.batch_size)

    print(
        f"Indexed {len(documents)} Tripletex endpoints into Azure AI Search index "
        f"'{settings.azure_search_index_name}'."
    )


def ensure_index(client: httpx.Client, settings: Any, recreate: bool) -> None:
    params = {"api-version": settings.azure_search_api_version}
    index_path = f"/indexes('{settings.azure_search_index_name}')"

    if recreate:
        delete_response = client.delete(index_path, params=params)
        if delete_response.status_code not in {204, 404}:
            delete_response.raise_for_status()

    response = client.get(index_path, params=params)
    if response.status_code == 200:
        return
    if response.status_code != 404:
        response.raise_for_status()

    definition = build_endpoint_index_definition(
        index_name=settings.azure_search_index_name,
        semantic_configuration=settings.azure_search_semantic_configuration,
    )
    create_response = client.put(
        index_path,
        params=params,
        headers={"Prefer": "return=representation"},
        json=definition,
    )
    create_response.raise_for_status()


def upload_documents(
    client: httpx.Client,
    settings: Any,
    documents: list[dict[str, Any]],
    batch_size: int,
) -> None:
    params = {"api-version": settings.azure_search_api_version}
    index_path = f"/indexes('{settings.azure_search_index_name}')/docs/search.index"

    for chunk in batched(documents, batch_size):
        response = client.post(index_path, params=params, json={"value": chunk})
        response.raise_for_status()
        body = response.json()
        failures = [item for item in body.get("value", []) if not item.get("status")]
        if failures:
            raise RuntimeError(f"Azure AI Search rejected {len(failures)} documents: {failures[:3]}")


def batched(items: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for index in range(0, len(items), batch_size):
        yield items[index:index + batch_size]


def extract_endpoint_documents(payload: Any) -> list[dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}

    if isinstance(payload, dict) and isinstance(payload.get("paths"), dict):
        tag_descriptions: dict[str, str] = {}
        for tag in payload.get("tags", []):
            if not isinstance(tag, dict):
                continue
            name = tag.get("name")
            description = tag.get("description", "")
            if isinstance(name, str):
                tag_descriptions[name] = str(description)
        server_url = ""
        servers = payload.get("servers", [])
        if isinstance(servers, list) and servers and isinstance(servers[0], dict):
            server_url = str(servers[0].get("url") or "")
        api_title = ""
        api_version = ""
        info = payload.get("info")
        if isinstance(info, dict):
            api_title = str(info.get("title") or "")
            api_version = str(info.get("version") or "")

        for path, path_item in payload["paths"].items():
            for document in extract_openapi_path_documents(
                path,
                path_item,
                tag_descriptions=tag_descriptions,
                server_url=server_url,
                api_title=api_title,
                api_version=api_version,
            ):
                documents[document["id"]] = document
        return list(documents.values())

    for candidate in walk_endpoint_records(payload):
        document = build_document_from_record(candidate)
        if document is not None:
            documents[document["id"]] = document

    return list(documents.values())


def extract_openapi_path_documents(
    path: str,
    path_item: Any,
    tag_descriptions: dict[str, str],
    server_url: str,
    api_title: str,
    api_version: str,
) -> list[dict[str, Any]]:
    if not isinstance(path_item, dict):
        return []

    shared_parameters = path_item.get("parameters")
    if not isinstance(shared_parameters, list):
        shared_parameters = []
    documents = []

    for method, operation in path_item.items():
        if method.lower() not in HTTP_METHODS or not isinstance(operation, dict):
            continue

        operation_parameters = operation.get("parameters")
        if not isinstance(operation_parameters, list):
            operation_parameters = []

        parameters = [*shared_parameters, *operation_parameters]
        record = {
            "path": path,
            "method": method.upper(),
            "operationId": operation.get("operationId"),
            "summary": operation.get("summary"),
            "description": operation.get("description"),
            "tags": operation.get("tags") or [],
            "tagDescriptions": [
                tag_descriptions.get(tag, "")
                for tag in operation.get("tags", [])
                if isinstance(tag, str) and tag_descriptions.get(tag)
            ],
            "parameters": parameters,
            "requestBody": operation.get("requestBody"),
            "responses": operation.get("responses"),
            "serverUrl": server_url,
            "apiTitle": api_title,
            "apiVersion": api_version,
        }
        document = build_document_from_record(record)
        if document is not None:
            documents.append(document)

    return documents


def walk_endpoint_records(node: Any) -> Iterable[dict[str, Any]]:
    if isinstance(node, list):
        for item in node:
            yield from walk_endpoint_records(item)
        return

    if not isinstance(node, dict):
        return

    if looks_like_endpoint_record(node):
        yield node

    for value in node.values():
        yield from walk_endpoint_records(value)


def looks_like_endpoint_record(record: dict[str, Any]) -> bool:
    path = record.get("path") or record.get("url") or record.get("endpoint")
    method = record.get("method") or record.get("httpMethod") or record.get("verb")
    text = record.get("description") or record.get("summary") or record.get("name") or record.get("operationId")
    return isinstance(path, str) and isinstance(method, str) and isinstance(text, str)


def build_document_from_record(record: dict[str, Any]) -> dict[str, Any] | None:
    path = first_string(record, "path", "url", "endpoint")
    method = first_string(record, "method", "httpMethod", "verb")
    if not path or not method:
        return None

    normalized_method = method.upper()
    operation_name = first_string(record, "operationId", "name", "title") or f"{normalized_method} {path}"
    summary = first_string(record, "summary", "shortDescription") or ""
    description = first_string(record, "description", "details") or summary
    tags = normalize_tags(record.get("tags"))
    tag_descriptions = [str(item) for item in record.get("tagDescriptions", []) if item]
    parameters = record.get("parameters") or record.get("params") or []
    request_body = record.get("requestBody") or record.get("body") or record.get("request")
    responses = record.get("responses") or record.get("response")
    server_url = first_string(record, "serverUrl") or ""
    api_title = first_string(record, "apiTitle") or ""
    api_version = first_string(record, "apiVersion") or ""

    content_parts = [
        f"Method: {normalized_method}",
        f"Path: {path}",
        f"Operation: {operation_name}",
    ]
    if summary:
        content_parts.append(f"Summary: {summary}")
    if description and description != summary:
        content_parts.append(f"Description: {description}")
    if tags:
        content_parts.append(f"Tags: {', '.join(tags)}")
    if tag_descriptions:
        content_parts.append("Tag descriptions: " + " | ".join(tag_descriptions))
    if server_url:
        content_parts.append(f"Server URL: {server_url}")
    if api_title:
        content_parts.append(f"API title: {api_title}")
    if api_version:
        content_parts.append(f"API version: {api_version}")
    if parameters:
        content_parts.append("Parameters: " + compact_json(parameters))
    if request_body is not None:
        content_parts.append("Request body: " + compact_json(request_body))
    if responses is not None:
        content_parts.append("Responses: " + compact_json(responses))

    return {
        "@search.action": "mergeOrUpload",
        "id": stable_document_id(normalized_method, path),
        "path": path,
        "method": normalized_method,
        "operationName": operation_name,
        "summary": summary,
        "description": description,
        "tags": tags,
        "content": "\n".join(content_parts),
    }


def normalize_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if value:
        return [str(value)]
    return []


def first_string(record: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def stable_document_id(method: str, path: str) -> str:
    digest = hashlib.sha1(f"{method}:{path}".encode("utf-8")).hexdigest()
    return digest[:40]


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    main()