import json
import os
import urllib.request


# ============================================================
# CONFIGURATION
# ============================================================

URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "nvidia/nemotron-3.5-lightning:free"

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY environment variable is not set.")


# ============================================================
# PYTHON TOOLS
# ============================================================

def calculate(a, b):
    return a + b


def divide(a, b):
    return a / b


# ============================================================
# TOOL DEFINITIONS
#
# Each tool contains:
#   - name
#   - description
#   - parameters (schema for the LLM)
#   - actual Python function
# ============================================================

tool_definitions = [

    {
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
        },
        "function": calculate
    },

    {
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
        },
        "function": divide
    }
]


# ============================================================
# GENERATE TOOL SCHEMAS FOR THE LLM
#
# We don't send the Python function itself.
# We only send the information the LLM needs to know
# about the tool.
# ============================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"]
        }
    }
    for tool in tool_definitions
]


# ============================================================
# TOOL REGISTRY
#
# Maps the tool name returned by the LLM
# to the actual Python function.
# ============================================================

tool_registry = {
    tool["name"]: tool["function"]
    for tool in tool_definitions
}


# ============================================================
# USER MESSAGE
# ============================================================

messages = [
    {
        "role": "user",
        "content": "I want to do calcualtin -- need to find 2 to the power 6 , then divide it by 4 , the number then multiplies by 68 and then divide that by 2 and if the result is greater than 100 then add 240 to it"
    }
]


# ============================================================
# LLM REQUEST 1
# ============================================================

payload = {
    "model": MODEL,
    "messages": messages,
    "tools": tools,
    "stream": False
}

print("\n========== LLM REQUEST 1 ==========")
print(json.dumps(payload, indent=2))


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


with urllib.request.urlopen(request) as response:
    result = json.loads(
        response.read().decode("utf-8")
    )


# ============================================================
# LLM RESPONSE 1
# ============================================================

print("\n========== LLM RESPONSE 1 ==========")
print(json.dumps(result, indent=2))


# ============================================================
# TOKEN USAGE 1
# ============================================================

usage = result.get("usage", {})

print("\n========== TOKEN USAGE 1 ==========")
print("Prompt tokens:", usage.get("prompt_tokens"))
print("Completion tokens:", usage.get("completion_tokens"))
print("Total tokens:", usage.get("total_tokens"))


# ============================================================
# PROCESS TOOL CALL
# ============================================================

message = result["choices"][0]["message"]

tool_calls = message.get("tool_calls", [])


if tool_calls:

    tool_call = tool_calls[0]

    function_name = tool_call["function"]["name"]

    arguments = json.loads(
        tool_call["function"]["arguments"]
    )

    print("LLM selected tool:", function_name)
    print("Arguments:", arguments)


    # --------------------------------------------------------
    # Find the Python function using the registry
    # --------------------------------------------------------

    tool_function = tool_registry.get(function_name)

    if tool_function is None:
        raise ValueError(
            f"Unknown tool: {function_name}"
        )


    # --------------------------------------------------------
    # Execute the selected Python function
    # --------------------------------------------------------

    tool_result = tool_function(**arguments)

    print("Tool result:", tool_result)


    # ========================================================
    # ADD TOOL CALL + RESULT TO MESSAGE HISTORY
    # ========================================================

    messages.append(message)

    messages.append({
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "content": str(tool_result)
    })


    # ========================================================
    # LLM REQUEST 2
    # ========================================================

    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": tools,
        "stream": False
    }

    print("\n========== LLM REQUEST 2 ==========")
    print(json.dumps(payload, indent=2))


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


    with urllib.request.urlopen(request) as response:
        final_result = json.loads(
            response.read().decode("utf-8")
        )


    # ========================================================
    # LLM RESPONSE 2
    # ========================================================

    print("\n========== LLM RESPONSE 2 ==========")
    print(json.dumps(final_result, indent=2))


    # ========================================================
    # TOKEN USAGE 2
    # ========================================================

    usage = final_result.get("usage", {})

    print("\n========== TOKEN USAGE 2 ==========")
    print("Prompt tokens:", usage.get("prompt_tokens"))
    print("Completion tokens:", usage.get("completion_tokens"))
    print("Total tokens:", usage.get("total_tokens"))


    # ========================================================
    # FINAL ANSWER
    # ========================================================

    final_answer = final_result["choices"][0]["message"]["content"]

    print("\nFinal answer:", final_answer)


else:

    # ========================================================
    # NO TOOL WAS SELECTED
    # ========================================================

    print("\nLLM answered directly:")

    print(message.get("content"))