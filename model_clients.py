import json
import urllib.error
import urllib.request

from typing import Dict, List, Self, Sequence, Tuple


class FakeModelClient:
    def __init__(self, outputs: Sequence):
        self.outputs = list(outputs)
        self.prompts = []

    def complete(self, prompt: str, max_new_tokens: int):
        self.prompts.append(prompt)
        if not self.outputs:
            raise RuntimeError("fake model ran out of outputs")
        return self.outputs.pop(0)


class LlamaCppModelClient:
    def __build_messages(
        self: Self, messages: List[Tuple[str, str]]
    ) -> List[Dict[str, str]]:
        return [{"role": role, "content": content} for role, content in messages]

    def __make_request(self: Self, request):
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"LlamaCpp request failed with HTTP {exc.code}: {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not reach LlamaCpp.\n"
                "Make sure `llama-server` is running.\n"
                f"URL: {self.url}\n"
                f"Model: {self.model}"
            ) from exc

    def __check_model(self: Self):
        request = urllib.request.Request(
            self.url + "/v1/models",
            headers={"Content-Type": "application/json"},
            method="GET",
        )
        data = self.__make_request(request)
        if data.get("error"):
            raise RuntimeError(f"Llama-server error: {data['error']}")

        idx = 0
        for t_idx, model in enumerate(data["models"]):
            if model["name"] == self.model:
                idx = t_idx
                break
        self.model = data["models"][idx]["name"]
        self.ctx = data["data"][idx]["meta"]["n_ctx"]

    def __init__(
        self: Self,
        model: str,
        host: str,
        port: int,
        temperature: float,
        top_p: float,
        timeout: int,
    ):
        self.url = f"http://{host}:{port}"
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.model = model
        self.__check_model()

    def complete(self, system: str, prompt: str, max_new_tokens: int) -> str:
        messages = [("system", system), ("user", prompt)]

        payload = {
            "model": self.model,
            "messages": self.__build_messages(messages),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_new_tokens,
        }
        request = urllib.request.Request(
            self.url + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        has_more_data = True
        assistant_message = ""
        while has_more_data:
            data = self.__make_request(request)
            with open("requests.txt", "w+") as f:
                f.write(json.dumps(payload["messages"]))
                f.write("\n\n")

            assistant_message = data["choices"][0]["message"]["content"]
            if data["choices"][0]["finish_reason"] != "length":
                has_more_data = False
                with open("answers.txt", "w+") as f:
                    f.write(json.dumps(data))
            else:
                payload["messages"] = self.__build_messages(
                    messages + [("assistant", assistant_message)]
                )
                request = urllib.request.Request(
                    self.url + "/v1/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )

        return assistant_message
