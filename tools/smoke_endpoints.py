from __future__ import annotations

import json
import sys
import urllib.request


def request(method: str, url: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def status(method: str, url: str) -> int:
    req = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(req, timeout=10) as response:
        return response.status


def main() -> None:
    base = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    health_head = status("HEAD", f"{base}/v1/healthz")
    health = request("GET", f"{base}/v1/healthz")
    metadata = request("GET", f"{base}/v1/metadata")
    context = request(
        "POST",
        f"{base}/v1/context",
        {
            "scope": "category",
            "context_id": "smoke",
            "version": 1,
            "payload": {"slug": "smoke", "display_name": "Smoke"},
        },
    )
    tick = request("POST", f"{base}/v1/tick", {"available_triggers": []})
    reply = request(
        "POST",
        f"{base}/v1/reply",
        {
            "conversation_id": "smoke-convo",
            "from_role": "merchant",
            "message": "ok",
        },
    )
    assert health_head == 200
    assert health["status"] == "ok"
    assert "model" in metadata
    assert context["accepted"] is True
    assert tick["actions"] == []
    assert reply["action"] in {"send", "wait", "end"}
    print("smoke passed")


if __name__ == "__main__":
    main()
