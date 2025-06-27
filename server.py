"""
PyAutoGUI MCP Server

This server provides tools for GUI automation using pyautogui:
- Click: Click at screen coordinates using percentage values
- Keypress: Send keyboard input to the active window
"""

import asyncio
import pyautogui
from typing import Any
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("pyautogui-server")

# Disable pyautogui fail-safe for automation (can be re-enabled for safety)
# pyautogui.FAILSAFE = True  # Uncomment for safety - move mouse to corner to abort

# Add supported keys list using PyAutoGUI's built-in KEYBOARD_KEYS
SUPPORTED_KEYS = pyautogui.KEYBOARD_KEYS

@mcp.tool()
async def click(x_percent: float, y_percent: float, button: str = "left", clicks: int = 1, interval: float = 0.0) -> str:
    """Click at a screen position using percentage coordinates.
    
    Args:
        x_percent: X coordinate as percentage of screen width (0.0 to 1.0)
        y_percent: Y coordinate as percentage of screen height (0.0 to 1.0)
        button: Mouse button to click ('left', 'right', 'middle')
        clicks: Number of clicks to perform
        interval: Time interval between clicks in seconds
    
    Returns:
        Success message with actual coordinates clicked
    """
    try:
        # Validate percentage inputs
        if not (0.0 <= x_percent <= 1.0) or not (0.0 <= y_percent <= 1.0):
            return "Error: Coordinates must be between 0.0 and 1.0 (percentage values)"
        
        # Get screen dimensions
        screen_width, screen_height = pyautogui.size()
        
        # Convert percentage to actual coordinates
        x = int(x_percent * screen_width)
        y = int(y_percent * screen_height)
        
        # Perform the click
        pyautogui.click(x, y, clicks=clicks, interval=interval, button=button)
        
        return f"Successfully clicked at ({x}, {y}) - {x_percent*100:.1f}%, {y_percent*100:.1f}% of screen with {button} button {clicks} time(s)"
        
    except Exception as e:
        return f"Error performing click: {str(e)}"

@mcp.tool()
async def keypress(keys: str, interval: float = 0.0) -> str:
    """Send keyboard input to the active window using pyautogui."""
    try:
        k = keys.lower().strip()
        # Normalize arrow key names (e.g., 'arrowleft' -> 'left')
        if k.startswith('arrow') and k[5:] in SUPPORTED_KEYS:
            k = k[5:]

        # Handle key combinations (e.g., 'ctrl+c')
        if '+' in k:
            parts = [p.strip() for p in k.split('+')]
            invalid = [p for p in parts if p not in SUPPORTED_KEYS]
            if invalid:
                return f"Error: Invalid key(s) {invalid}. Supported keys: {sorted(SUPPORTED_KEYS)}"
            pyautogui.hotkey(*parts)
            return f"Sent key combination: {keys}"

        # Handle multi-key sequences separated by spaces (e.g., "t r i c k enter")
        if ' ' in k:
            parts = [p.strip() for p in k.split()]
            invalid = [p for p in parts if p not in SUPPORTED_KEYS]
            if invalid:
                return f"Error: Invalid key(s) {invalid}. Supported keys: {sorted(SUPPORTED_KEYS)}"
            # Press each key in sequence
            for p in parts:
                pyautogui.press(p, interval=interval)
            return f"Sent key sequence: {keys}"

        # Handle single supported key press
        if k in SUPPORTED_KEYS:
            pyautogui.press(k, interval=interval)
            return f"Sent key: {keys}"

        # Handle concatenated key names without delimiters (e.g., 'backspacebackspace')
        seq = []
        remaining = k
        # Try to consume known keys greedily by descending length
        for key_name in sorted(SUPPORTED_KEYS, key=len, reverse=True):
            while remaining.startswith(key_name):
                seq.append(key_name)
                remaining = remaining[len(key_name):]
        if seq and not remaining:
            for p in seq:
                pyautogui.press(p, interval=interval)
            return f"Sent key sequence: {keys}"

        # Fallback to typing any text
        pyautogui.write(keys, interval=interval)
        return f"Typed text: {keys}"
    except Exception as e:
        return f"Error performing keypress: {str(e)}"

@mcp.tool()
async def get_screen_info() -> str:
    """Get information about the screen dimensions and current mouse position.
    
    Returns:
        Screen dimensions and current mouse position information
    """
    try:
        screen_width, screen_height = pyautogui.size()
        mouse_x, mouse_y = pyautogui.position()
        
        # Calculate mouse position as percentages
        mouse_x_percent = mouse_x / screen_width
        mouse_y_percent = mouse_y / screen_height
        
        return f"""Screen Information:
- Screen Size: {screen_width} x {screen_height} pixels
- Current Mouse Position: ({mouse_x}, {mouse_y})
- Mouse Position (percentage): {mouse_x_percent:.3f}, {mouse_y_percent:.3f}
- To click at current mouse position, use: click({mouse_x_percent:.3f}, {mouse_y_percent:.3f})"""
        
    except Exception as e:
        return f"Error getting screen info: {str(e)}"

if __name__ == "__main__":
    print("Starting PyAutoGUI MCP Server...")
    print("Available tools:")
    print("- click: Click at screen coordinates using percentage values")
    print("- keypress: Send keyboard input")
    print("- get_screen_info: Get screen dimensions and mouse position")
    print("- focus_window: Click at the center of the screen to focus the window")
    
    # Run the server
    mcp.run(transport="stdio")
