import os
import requests

response = requests.post(
    "https://usa0.top/v1/messages",
    headers={
        "x-api-key": os.environ["ANTHROPIC_AUTH_TOKEN"],
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    json={
        "model": "claude-haiku-5-5",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "用一段话解释量子纠缠。"}
        ],
    },
    timeout=60,
)

print("HTTP 状态码:", response.status_code)
print(response.text)
