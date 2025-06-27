"""
Test script for PyAutoGUI MCP Server

This script demonstrates the basic functionality of the PyAutoGUI tools
without requiring the full MCP client setup.
"""

import asyncio
import sys
import os

# Add the current directory to the path so we can import the server
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server import click, keypress, get_screen_info, screenshot

async def test_tools():
    """Test all PyAutoGUI tools."""
    print("🧪 Testing PyAutoGUI MCP Tools")
    print("=" * 40)
    
    # Test screen info
    print("\n1. Testing get_screen_info...")
    screen_info = await get_screen_info()
    print(screen_info)
    
    # Test keypress
    print("\n3. Testing keypress (will type 'test' in active window)...")
    print("⚠️  Make sure a text editor or notepad is active!")
    await asyncio.sleep(3)  # Give user time to switch windows
    
    keypress_result = await keypress("test")
    print(keypress_result)
    
    # Test click (click at center of screen)
    print("\n4. Testing click (center of screen)...")
    print("⚠️  This will click at the center of your screen in 3 seconds!")
    await asyncio.sleep(3)
    
    click_result = await click(0.5, 0.5)
    print(click_result)
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    print("This will test the PyAutoGUI tools directly.")
    print("Make sure you have a text editor open for the keypress test.")
    
    response = input("Continue? (y/n): ")
    if response.lower() == 'y':
        asyncio.run(test_tools())
    else:
        print("Test cancelled.")
