import json
import urllib.request

URL = "http://localhost:11434/api/chat"

messages = [
    {
        "role": "system",
        "content": "Answer questions accurately using the conversation."
    }
]

def chat(user_message):
    # Add the user's message to the conversation
    messages.append({
        "role": "user",
        "content": user_message
    })

    payload = {
        "model": "tinyllama:1.1b-chat",
        "messages": messages,
        "stream": False
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read().decode("utf-8"))

    assistant_message = result["message"]["content"]

    # Store the assistant's response too
    messages.append({
        "role": "assistant",
        "content": assistant_message
    })

    return assistant_message


print("First response:")
print(chat("My name is Sunath. Remember my name."))

print("\nSecond response:")
print(chat("What is my name?"))