import json
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def post_chat_completion(
    base_url,
    model,
    messages,
    temperature,
    max_tokens,
    timeout,
    *,
    top_p=1.0,
    enable_thinking=False,
    extra_body=None,
):
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if extra_body:
        body.update(extra_body)
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


def chat_with_retries(
    *,
    base_url,
    model,
    messages,
    temperature,
    max_tokens,
    timeout_seconds,
    retries,
    parser=None,
    top_p=1.0,
    enable_thinking=False,
    extra_body=None,
):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            content = post_chat_completion(
                base_url,
                model,
                messages,
                temperature,
                max_tokens,
                timeout_seconds,
                top_p=top_p,
                enable_thinking=enable_thinking,
                extra_body=extra_body,
            )
            parsed = parser(content) if parser else content
            return content, parsed, attempt
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            time.sleep(min(2 * attempt, 8))
    raise RuntimeError(f"Chat completion failed after {retries} attempts: {last_error}")


def extract_json(text):
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(stripped[start : end + 1])
