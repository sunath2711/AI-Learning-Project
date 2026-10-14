import sys

from mcp.server.fastmcp import FastMCP

#create an mcp server named Calculator
mcp = FastMCP("Calculator")

@mcp.tool()
def calculate(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide the first number by the second number."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

if __name__ == "__main__":
    print("Starting MCP server...", file=sys.stderr, flush=True)
    mcp.run(transport="stdio")