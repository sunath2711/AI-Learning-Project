import json
import os
import urllib.request


URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "nvidia/nemotron-3.5-lightning:free"

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY environment variable is not set."
    )


# -----------------------------
# Our Python tool
# -----------------------------

def calculate(a, b):
    return a + b

def divide(a, b):
    return a / b
# -----------------------------
# Tool description for the LLM
# -----------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Add two numbers together.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "First number"
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number"
                    }
                },
                "required": ["a", "b"]
            }
        }
    }
    ,
    {
    "type": "function",
    "function": {
        "name": "divide",
        "description": "Divide the first number by the second number.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "Number to be divided"
                },
                "b": {
                    "type": "number",
                    "description": "Number to divide by"
                }
            },
            "required": ["a", "b"]
        }
    }
}
]


# -----------------------------
# Ask the LLM
# -----------------------------

# messages = [
#     {
#         "role": "user",
#         "content": "Can you add 25  and 17?"
#     }
# ]

# messages = [
#     {
#         "role": "user",
#         "content": "Can you divide 100 and 25 and then add 1000 to it?"
#     }
# ]
messages = [
    {
        "role": "user",
        "content": "I want to perform a division on 2 binary numbers 100000000 and 1000?"
    }
]

payload = {
    "model": MODEL,
    "messages": messages,
    "tools": tools,
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
print("\n========== LLM REQUEST 1 ==========")
print(json.dumps(payload, indent=2))

with urllib.request.urlopen(request) as response:
    result = json.loads(
        response.read().decode("utf-8")
    )
print("\n========== LLM RESPONSE 1 ==========")
print(json.dumps(result, indent=2))

usage = result.get("usage", {})

print("\n========== TOKEN USAGE 1 ==========")
print("Prompt tokens:", usage.get("prompt_tokens"))
print("Completion tokens:", usage.get("completion_tokens"))
print("Total tokens:", usage.get("total_tokens"))

# print(json.dumps(result, indent=2))
message = result["choices"][0]["message"]

tool_calls = message.get("tool_calls", [])

if tool_calls:
    tool_call = tool_calls[0]

    function_name = tool_call["function"]["name"]
    arguments = json.loads(tool_call["function"]["arguments"])

    print("LLM selected tool:", function_name)
    print("Arguments:", arguments)

    if function_name == "calculate":
        tool_result = calculate(**arguments)
    elif function_name == "divide":
        tool_result = divide(**arguments)
    else:
        raise ValueError(f"Unknown tool: {function_name}")

    print("Tool result:", tool_result)

    messages.append(message)

    messages.append({
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "content": str(tool_result)
    })

    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": tools,
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

    print("\n========== LLM REQUEST 2 ==========")
    print(json.dumps(payload, indent=2))
    with urllib.request.urlopen(request) as response:
        final_result = json.loads(
            response.read().decode("utf-8")
        )

    final_answer = final_result["choices"][0]["message"]["content"]

    print("\n========== LLM RESPONSE 2 ==========")
    print(json.dumps(final_result, indent=2))

    usage = final_result.get("usage", {})

    print("\n========== TOKEN USAGE 2 ==========")
    print("Prompt tokens:", usage.get("prompt_tokens"))
    print("Completion tokens:", usage.get("completion_tokens"))
    print("Total tokens:", usage.get("total_tokens"))

    print("Final answer:", final_answer)
