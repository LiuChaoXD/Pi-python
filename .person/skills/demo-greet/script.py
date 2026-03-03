#!/usr/bin/env python3
"""
demo-greet - A simple demo skill that greets the user. It reads a JSON object with a 'name' field from stdin and returns a greeting message.

Auto-generated skill script.
Input: JSON params via stdin
Output: JSON result via stdout
"""

import json
import sys

def main():
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        data = {}

    # Extract the 'name' field, defaulting to 'World' if not provided
    name = data.get("name", "World")

    # Create the greeting result
    result = {"greeting": f"Hello, {name}!"}

    # Output the result as JSON
    print(json.dumps(result))

if __name__ == "__main__":
    main()
