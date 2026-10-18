
import asyncio
import json
import subprocess
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


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

            # 3. Initialize the MCP connection
            await session.initialize()
            print("Connected to MCP server!")
            #This establishes the MCP session and performs the initialization handshake between the client and server.

            # 4. Discover available tools
            response = await session.list_tools()

            # print("\nAvailable tools:")
            # for tool in response.tools:
            #     print(tool.inputSchema)
            #     print(f"- {tool.name}: {tool.description}")
            print("\nAvailable tools:")

            llm_tools = []

            for tool in response.tools:
                print(f"- {tool.name}: {tool.description}")

                llm_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema
                    }
                })

            print("\nTool schemas for the LLM:")
            print(json.dumps(llm_tools, indent=2))

            # 5. Ask the server to execute a tool
            result = await session.call_tool(
                "divide",
                arguments={"a": 100, "b": 5}
            )

            print("\nTool result:")
            for item in result.content:
                if item.type == "text":
                    print(item.text)


if __name__ == "__main__":
    asyncio.run(main())
