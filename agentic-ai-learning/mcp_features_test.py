
import asyncio
import subprocess
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    project_root = Path(__file__).resolve().parent

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(project_root / "mcp_server.py")],
        cwd=str(project_root),
    )

    async with stdio_client(
        server_params, errlog=subprocess.DEVNULL
    ) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Connected to MCP server!")

            # 1. List resources
            resources = await session.list_resources()
            print("\nAvailable resources:")
            for resource in resources.resources:
                print(f"- {resource.name}: {resource.uri}")

            # 2. Read the calculator resource
            resource_result = await session.read_resource(
                "info://calculator"
            )
            print("\nResource contents:")
            for item in resource_result.contents:
                if hasattr(item, "text"):
                    print(item.text)

            # 3. List prompts
            prompts = await session.list_prompts()
            print("\nAvailable prompts:")
            for prompt in prompts.prompts:
                print(f"- {prompt.name}: {prompt.description}")

            # 4. Retrieve a prompt with an argument
            prompt_result = await session.get_prompt(
                "solve_math_problem",
                arguments={
                    "problem": "Find the square of (1000 % 4) * (90 + 80)"
                },
            )
            print("\nGenerated prompt:")
            for message in prompt_result.messages:
                print(message.content.text)


if __name__ == "__main__":
    asyncio.run(main())
