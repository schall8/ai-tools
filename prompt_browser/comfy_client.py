"""Minimal ComfyUI API client: submit a workflow graph, wait, fetch images."""
import json
import uuid
import urllib.parse

import requests
from websocket import create_connection


class ComfyClient:
    def __init__(self, base_url="http://127.0.0.1:8188"):
        self.base = base_url.rstrip("/")
        self.ws_base = self.base.replace("https://", "wss://").replace("http://", "ws://")
        self.client_id = str(uuid.uuid4())

    def queue(self, graph):
        """POST a workflow (API format) -> prompt_id."""
        r = requests.post(
            "%s/prompt" % self.base,
            json={"prompt": graph, "client_id": self.client_id},
            timeout=30,
        )
        if r.status_code != 200:
            # ComfyUI returns validation errors in the body -- surface them.
            raise RuntimeError("ComfyUI rejected the prompt (%d): %s" % (r.status_code, r.text[:1000]))
        return r.json()["prompt_id"]

    def wait(self, prompt_id, on_progress=None, timeout=600):
        """Block until the given prompt finishes, streaming progress."""
        ws = create_connection("%s/ws?clientId=%s" % (self.ws_base, self.client_id), timeout=timeout)
        try:
            while True:
                msg = ws.recv()
                if isinstance(msg, (bytes, bytearray)):
                    continue  # binary preview frames
                data = json.loads(msg)
                mtype, mdata = data.get("type"), data.get("data", {})
                if mtype == "progress" and on_progress:
                    on_progress(mdata.get("value", 0), mdata.get("max", 1))
                if (mtype == "executing" and mdata.get("node") is None
                        and mdata.get("prompt_id") == prompt_id):
                    break  # this prompt is done
        finally:
            ws.close()

    def history(self, prompt_id):
        return requests.get("%s/history/%s" % (self.base, prompt_id), timeout=30).json()

    def images(self, prompt_id):
        """Return [(filename, bytes), ...] for the finished prompt."""
        hist = self.history(prompt_id).get(prompt_id, {})
        out = []
        for node_out in hist.get("outputs", {}).values():
            for img in node_out.get("images", []):
                q = urllib.parse.urlencode(img)
                data = requests.get("%s/view?%s" % (self.base, q), timeout=60).content
                out.append((img.get("filename"), data))
        return out

    def generate(self, graph, on_progress=None):
        pid = self.queue(graph)
        self.wait(pid, on_progress)
        return self.images(pid)
