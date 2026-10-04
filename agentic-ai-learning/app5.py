import json
import os
import urllib.request
import urllib.error

from memory_extractor3 import (
    extract_memory,
    load_existing_memory,
    merge_memory,
    save_memory
)


URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "nvidia/nemotron-3.5-lightning:free"

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY environment variable is not set."
    )


MEMORY_FILE = "memory.json"


# Load previous conversation
if os.path.exists(MEMORY_FILE):

    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        messages = json.load(f)

else:

    messages = [
        {
            "role": "system",
            "content": "You are a helpful and accurate assistant."
        }
    ]


# Load long-term user memory
user_memory = load_existing_memory()


def save_conversation():

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(
            messages,
            f,
            indent=2,
            ensure_ascii=False
        )


def update_user_memory(conversation):

    global user_memory

    new_memory = extract_memory(conversation)

    merged_memory = merge_memory(
        user_memory,
        new_memory
    )

    save_memory(merged_memory)

    user_memory = merged_memory


def chat(user_message):

    messages.append({
        "role": "user",
        "content": user_message
    })


    # Short-term conversation window
    recent_messages = messages[:1] + messages[-6:]


    # Add long-term memory
    memory_message = {
        "role": "system",
        "content": (
            "Long-term user memory: "
            f"{json.dumps(user_memory)}"
        )
    }

    recent_messages = (
        [recent_messages[0], memory_message]
        + recent_messages[1:]
    )


    # Main LLM request
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

            result = json.loads(
                response.read().decode("utf-8")
            )

    except urllib.error.HTTPError as e:

        print("HTTP Status:", e.code)
        print(
            "OpenRouter Error:",
            e.read().decode("utf-8")
        )
        raise


    print("Token usage:", result.get("usage"))


    if "choices" not in result:

        print("Unexpected API response:")
        print(json.dumps(result, indent=2))

        raise RuntimeError(
            "OpenRouter did not return a normal completion response."
        )


    assistant_message = (
        result["choices"][0]["message"]["content"]
    )


    messages.append({
        "role": "assistant",
        "content": assistant_message
    })


    save_conversation()


    # ------------------------------------------
    # LONG-TERM MEMORY UPDATE
    # ------------------------------------------

    memory_conversation = messages[-6:]

    conversation_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in memory_conversation
    )

    update_user_memory(conversation_text)


    return assistant_message


while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        break

    response = chat(user_input)

    print("Assistant:", response)