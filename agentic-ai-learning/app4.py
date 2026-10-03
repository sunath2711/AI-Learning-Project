import json
import os
import urllib.request
import urllib.error


URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "nvidia/nemotron-3.5-lightning:free"

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY environment variable is not set.")


# messages = [
#     {
#         "role": "system",
#         "content": "You are a helpful and accurate assistant."
#     }
# ]


# def chat(user_message):

#     messages.append({
#         "role": "user",
#         "content": user_message
#     })

#     payload = {
#         "model": MODEL,
#         "messages": messages,
#         "stream": False
#     }

#     data = json.dumps(payload).encode("utf-8")

#     request = urllib.request.Request(
#         URL,
#         data=data,
#         headers={
#             "Content-Type": "application/json",
#             "Authorization": f"Bearer {API_KEY}"
#         },
#         method="POST"
#     )

#     try:
#         with urllib.request.urlopen(request) as response:
#             result = json.loads(response.read().decode("utf-8"))

#     except urllib.error.HTTPError as e:
#         print("HTTP Status:", e.code)
#         print("OpenRouter Error:", e.read().decode("utf-8"))
#         raise

#     assistant_message = result["choices"][0]["message"]["content"]

#     messages.append({
#         "role": "assistant",
#         "content": assistant_message
#     })

#     return assistant_message


# while True:

#     user_input = input("\nYou: ")

#     if user_input.lower() == "exit":
#         break

#     response = chat(user_input)

#     print("Assistant:", response)

MEMORY_FILE = "memory.json"
USER_MEMORY_FILE = "user_memory.json"

# Load previous conversation if it exists
if os.path.exists(MEMORY_FILE):

    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        messages = json.load(f)

    with open(USER_MEMORY_FILE, "r", encoding="utf-8") as f:
        user_memory = json.load(f)

else:

    messages = [
        {
            "role": "system",
            "content": "You are a helpful and accurate assistant."
        }
    ]


def save_memory():

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)


def chat(user_message):

    messages.append({
        "role": "user",
        "content": user_message
    })
    recent_messages = messages[:1] + messages[-6:]
    memory_message = {
    "role": "system",
    "content": f"Long-term user memory: {json.dumps(user_memory)}"
}

    recent_messages = [recent_messages[0], memory_message] + recent_messages[1:]

    payload = {
        "model": MODEL,
        "messages": recent_messages,
        "stream": False
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        },
        method="POST"
    )

    try:

        with urllib.request.urlopen(request) as response:
            result = json.loads(response.read().decode("utf-8"))

    except urllib.error.HTTPError as e:

        print("HTTP Status:", e.code)
        print("OpenRouter Error:", e.read().decode("utf-8"))
        raise
    print("Token usage:", result.get("usage"))
    if "choices" not in result:
        print("Unexpected API response:")
        print(json.dumps(result, indent=2))
        raise RuntimeError("OpenRouter did not return a normal completion response.")
    assistant_message = result["choices"][0]["message"]["content"]

    messages.append({
        "role": "assistant",
        "content": assistant_message
    })

    save_memory()

    return assistant_message


while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        break

    response = chat(user_input)

    print("Assistant:", response)
