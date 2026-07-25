from ollama import chat

prompt = """
Return ONLY valid JSON.

[
  {
    "name": "Python",
    "score": 100
  },
  {
    "name": "SQL",
    "score": 95
  }
]
"""

response = chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "system",
            "content": "You are a JSON generator. Respond ONLY with valid JSON. Do not explain."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print(response["message"]["content"])