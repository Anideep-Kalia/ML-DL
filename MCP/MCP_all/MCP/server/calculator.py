# calc_server.py
from mcp.server import Server

server = Server("calc-server")

@server.tool("add")
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@server.tool("subtract")
def subtract(a: float, b: float) -> float:
    """Subtract two numbers."""
    return a - b

@server.tool("multiply")
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

@server.tool("divide")
def divide(a: float, b: float) -> float:
    """Divide two numbers (b must not be 0)."""
    if b == 0:
        return float("inf")
    return a / b

if __name__ == "__main__":
    server.run()
