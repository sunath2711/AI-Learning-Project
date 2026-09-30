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

    try:
        with urllib.request.urlopen(request) as response:
            result = json.loads(response.read().decode("utf-8"))

    except urllib.error.HTTPError as e:
        print("HTTP Status:", e.code)
        print("Ollama Error:", e.read().decode("utf-8"))
        raise

    assistant_message = result["message"]["content"]

    # Store the assistant's response too
    messages.append({
        "role": "assistant",
        "content": assistant_message
    })

    return assistant_message


while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        break

    response = chat(user_input)

    print("Assistant:", response)