import requests
from typing import Any, Dict, List, Optional


class APIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base = base_url.rstrip("/")

    def health(self) -> Dict[str, Any]:
        return requests.get(f"{self.base}/api/health").json()

    def list_documents(self) -> Dict[str, Any]:
        return requests.get(f"{self.base}/api/documents").json()

    def upload_document(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, "rb") as fh:
            files = {"file": (file_path.split("/")[-1], fh)}
            r = requests.post(f"{self.base}/api/documents/upload", files=files)
            r.raise_for_status()
            return r.json()

    def delete_document(self, document_id: str) -> Dict[str, Any]:
        r = requests.delete(f"{self.base}/api/documents/{document_id}")
        r.raise_for_status()
        return r.json()

    def reindex(self, document_id: str, file_path: str) -> Dict[str, Any]:
        r = requests.post(f"{self.base}/api/reindex/{document_id}", json={"file_path": file_path})
        r.raise_for_status()
        return r.json()

    def query(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        payload = {"query": query}
        if top_k is not None:
            payload["top_k"] = top_k
        r = requests.post(f"{self.base}/api/query", json=payload)
        r.raise_for_status()
        return r.json()

    def fetch_file_url(self, filename: str) -> str:
        return f"{self.base}/api/files/{filename}"
