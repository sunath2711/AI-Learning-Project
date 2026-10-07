import json
import os
import urllib.request



# ============================================================
# CONFIGURATION
# ============================================================

URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "nvidia/nemotron-3.5-lightning:free"
MAX_ITERATIONS = 10
iteration = 0

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
        #"content": "I want to do calcualtin -- need to find 2 to the power 6 , then divide it by 4 , the number then multiplies by 68 and then divide that by 2 and if the result is greater than 100 then add 240 to it"
        "content": "I want you to perfrom a subtraction of two prime numbers that lie between 400 and 900 and then multiple that by any perfect number, and then add 190 to it and then give the answer in word , also each step when u asnwer give me your reasoning"
    }
]


# ============================================================
# LLM REQUEST 1
# ============================================================

while iteration < MAX_ITERATIONS:
    iteration += 1
    print(f"\n========== AGENT ITERATION {iteration} ==========")
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

    with urllib.request.urlopen(request) as response:
        result = json.loads(
            response.read().decode("utf-8")
        )

    message = result["choices"][0]["message"]
    finish_reason = result["choices"][0]["finish_reason"]

    print("\nFinish reason:", finish_reason)

    # If the model wants to use a tool
    if finish_reason == "tool_calls":

        tool_calls = message.get("tool_calls", [])

        # Add the assistant's tool-call message to conversation
        messages.append(message)

        for tool_call in tool_calls:

            function_name = tool_call["function"]["name"]
            arguments = json.loads(
                tool_call["function"]["arguments"]
            )

            print("LLM selected tool:", function_name)
            print("Arguments:", arguments)

            tool_function = tool_registry.get(function_name)

            if tool_function is None:
                tool_content = (
                    f"Tool execution failed: unknown tool '{function_name}'"
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_content
                })

                continue
            try:
                tool_result = tool_function(**arguments)

                print("Tool result:", tool_result)

                tool_content = str(tool_result)

            except Exception as e:
                print("Tool error:", str(e))

                tool_content = f"Tool execution failed: {str(e)}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": tool_content
            })

        # Go back to the LLM
        continue

    
    # If the model has finished
    elif finish_reason == "stop":

        final_answer = message["content"]

        print("\nFinal answer:", final_answer)

        break


if iteration >= MAX_ITERATIONS:
        print("\nAgent stopped: maximum iterations reached.")
