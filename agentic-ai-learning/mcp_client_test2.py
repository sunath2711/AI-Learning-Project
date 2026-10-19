
import asyncio
import json
import subprocess
import sys
from pathlib import Path
import os
import urllib.request
import urllib.error
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def call_llm(payload):
    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is not set."
        )

    request = urllib.request.Request(
        url="https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.loads(response.read().decode("utf-8"))

    if "choices" not in result or not result["choices"]:
        error_details = result.get("error") or result
        raise RuntimeError(f"OpenRouter returned no choices: {json.dumps(error_details, indent=2)}")

    return result


def assistant_message_for_history(message):
    clean_message = {
        "role": "assistant",
        "content": message.get("content") or "",
    }

    tool_calls = message.get("tool_calls")
    if tool_calls:
        clean_message["tool_calls"] = tool_calls

    return clean_message

    
async def main():
    project_root = Path(__file__).resolve().parent

    # 1. Tell the client how to start the MCP server.
    # Use a clean stderr target so the child process does not inherit the interactive terminal
    # handle, which can break stdio MCP handshakes in this environment.
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(project_root / "mcp_server.py")],
        cwd=str(project_root),
    )

    # 2. Start the server and establish communication.
    # Redirect child stderr away from the terminal to prevent the stdio handshake from exiting.
    async with stdio_client(server_params, errlog=subprocess.DEVNULL) as (read, write):
        async with ClientSession(read, write) as session:

                await session.initialize()
                print("Connected to MCP server!")

            # 1. Discover tools from the MCP server
                response = await session.list_tools()

                # 2. Convert MCP tool definitions into the LLM API format
                llm_tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.inputSchema,
                        },
                    }
                    for tool in response.tools
                ]

                # 3. Ask the LLM to use a tool
                payload = {
                    "model": "nvidia/nemotron-3.5-lightning:free",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Pls help me find the answer for square of [(1000%4)*(90+80)] ."
                        }
                    ],
                    "tools": llm_tools,
                    "tool_choice": "auto",
                    # "tool_choice": "required",                    
                }

                # # 4. Send the request to OpenRouter
                # llm_response = call_llm(payload)

                # # 5. Inspect the LLM's response
                # choice = llm_response["choices"][0]
                # message = choice["message"]

                # print("LLM response:")
                # print(json.dumps(message, indent=2))


                # # Execute the tool if the LLM requested one
                # tool_calls = message.get("tool_calls", [])

                # for tool_call in tool_calls:
                #     function_name = tool_call["function"]["name"]

                #     arguments = json.loads(
                #         tool_call["function"]["arguments"]
                #     )

                #     print("\nExecuting MCP tool:", function_name)
                #     print("Arguments:", arguments)

                #     result = await session.call_tool(
                #         function_name,
                #         arguments=arguments,
                #     )

                #     print("MCP tool result:")

                #     for item in result.content:
                #         if item.type == "text":
                #             print(item.text)

                # tool_calls = message.get("tool_calls", [])

                # if tool_calls:
                #     # 1. Start with the original user message
                #     messages = payload["messages"].copy()

                #     # 2. Append the assistant message containing the tool call
                #     messages.append(message)

                #     # 3. Execute each requested MCP tool
                #     for tool_call in tool_calls:
                #         function_name = tool_call["function"]["name"]

                #         arguments = json.loads(
                #             tool_call["function"]["arguments"]
                #         )

                #         print("\nExecuting MCP tool:", function_name)
                #         print("Arguments:", arguments)

                #         result = await session.call_tool(
                #             function_name,
                #             arguments=arguments,
                #         )

                #         # Extract text returned by the MCP server
                #         tool_result = "\n".join(
                #             item.text
                #             for item in result.content
                #             if item.type == "text"
                #         )

                #         print("MCP tool result:", tool_result)

                #         # 4. Append the tool result with its matching call ID
                #         messages.append({
                #             "role": "tool",
                #             "tool_call_id": tool_call["id"],
                #             "content": tool_result,
                #         })
                
                messages = payload["messages"].copy()

                MAX_ITERATIONS = 8

                for iteration in range(MAX_ITERATIONS):
                    current_payload = {
                        "model": payload["model"],
                        "messages": messages,
                        "tools": llm_tools,
                        "tool_choice": "auto",
                    }

                    llm_response = call_llm(current_payload)
                    message = llm_response["choices"][0]["message"]

                    # Preserve only the chat-safe assistant fields in history.
                    # The model can return extra metadata such as reasoning/refusal,
                    # which is not valid to send back in follow-up messages.
                    messages.append(assistant_message_for_history(message))

                    tool_calls = message.get("tool_calls") or []

                    # No tool calls means the agent has its final answer.
                    if not tool_calls:
                        print("\nFinal answer:")
                        print(message.get("content"))
                        break

                    # Execute every tool requested in this response.
                    for tool_call in tool_calls:
                        function_name = tool_call["function"]["name"]
                        arguments = json.loads(
                            tool_call["function"]["arguments"]
                        )

                        print(f"\nExecuting MCP tool: {function_name}")
                        print("Arguments:", arguments)

                        result = await session.call_tool(
                            function_name,
                            arguments=arguments,
                        )

                        tool_result = "\n".join(
                            item.text
                            for item in result.content
                            if item.type == "text"
                        )

                        print("MCP tool result:", tool_result)

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": tool_result,
                        })

                else:
                    print("Stopped: maximum tool-call iterations reached.")
                    # # 5. Ask the LLM to respond using the tool result
                    # final_payload = {
                    #     "model": "nvidia/nemotron-3.5-lightning:free",
                    #     "messages": messages,
                    # }

                    # final_response = call_llm(final_payload)

                    # final_message = final_response["choices"][0]["message"]

                    # print("\nFinal answer:")
                    # print(final_message["content"])

if __name__ == "__main__":
    asyncio.run(main())
