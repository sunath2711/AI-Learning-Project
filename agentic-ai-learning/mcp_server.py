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

@mcp.resource("info://calculator")
def calculator_info() -> str:
    """Describe the calculator tools available on this server."""
    return (
        "Available operations:\n"
        "1. calculate(a, b): adds two numbers.\n"
        "2. divide(a, b): divides the first number by the second.\n"
        "Division by zero is not allowed."
    )

@mcp.prompt()
def solve_math_problem(problem: str) -> str:
    """Create a reusable prompt for solving a math problem."""
    return (
        "Solve the following math problem step by step.\n"
        "Use available calculator tools when appropriate.\n"
        f"Problem: {problem}"
    )

if __name__ == "__main__":
    print("Starting MCP server...", file=sys.stderr, flush=True)
    mcp.run(transport="stdio")