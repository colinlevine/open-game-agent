"""
Thought Overlay Window

A floating overlay window that displays AI agent thoughts in real-time.
Thread-safe implementation using tkinter with command queuing.
"""

import tkinter as tk
from tkinter import scrolledtext
import threading
import queue


class ThoughtOverlay:
    def __init__(self, width=600, height=None):
        self.width = width
        self.height = height  # if None, we'll fill vertical screen height
        self.content = "AI Agent Thoughts\nWaiting for content..."
        self.root = None
        self.text_widget = None
        self.running = False
        # overlay visibility state
        self._visible = True
        self._command_queue = queue.Queue()

    def hide(self):
        """Request to hide the overlay window safely"""
        self._command_queue.put('hide')

    def show(self):
        """Request to show the overlay window safely"""
        self._command_queue.put('show')

    def start(self):
        """Start the overlay in the main thread"""
        self.running = True
        threading.Thread(target=self._create_overlay, daemon=True).start()
        
    def _create_overlay(self):
        """Create the overlay window"""
        try:
            # Create root window
            self.root = tk.Tk()
            self.root.title("AI Agent Thoughts")
            
            # Make it always stay on top and prevent focus stealing
            self.root.attributes('-topmost', True)
            self.root.attributes('-alpha', 0.75)
            
            # Disable window decorations for cleaner look
            self.root.overrideredirect(True)
            
            # Determine size and position
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            # Fill full vertical height if height not set
            if self.height is None:
                # Reserve small top margin
                self.height = screen_height - 40
            x = screen_width - self.width - 20
            y = 20
            self.root.geometry(f"{self.width}x{self.height}+{x}+{y}")
            
            # Prevent window from being moved or resized
            self.root.resizable(False, False)
            
            # Create main frame with border
            main_frame = tk.Frame(self.root, bg='black', highlightbackground='white', highlightthickness=2)
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Title label
            title_label = tk.Label(main_frame, text="AI Agent Thoughts", 
                                 bg='black', fg='white', font=('Arial', 12, 'bold'))
            title_label.pack(pady=(5, 0))
            
            # Separator
            separator = tk.Frame(main_frame, height=2, bg='white')
            separator.pack(fill=tk.X, padx=10, pady=5)
            
            # Text area
            self.text_widget = scrolledtext.ScrolledText(
                main_frame,
                wrap=tk.WORD,
                bg='black',
                fg='white',
                font=('Consolas', 9),
                insertbackground='white',
                highlightthickness=0,
                bd=0
            )
            self.text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
            
            # Insert initial content
            self.text_widget.insert('1.0', self.content)
            
            # Force window to stay in position
            def keep_on_top():
                if self.running and self.root:
                    try:
                        self.root.attributes('-topmost', True)
                        # Reset position if it moved
                        current_geo = self.root.geometry()
                        if not current_geo.endswith(f"+{x}+{y}"):
                            self.root.geometry(f"{self.width}x{self.height}+{x}+{y}")
                    except:
                        pass
            
            # Start the keep-on-top loop
            self.root.after(100, keep_on_top)
            
            # Prevent the window from being destroyed accidentally
            self.root.protocol("WM_DELETE_WINDOW", lambda: None)
            # Process pending hide/show commands in the GUI thread
            def poll_queue():
                try:
                    cmd = self._command_queue.get_nowait()
                    if cmd == 'hide':
                        self.root.withdraw()
                    elif cmd == 'show':
                        self.root.deiconify()
                except queue.Empty:
                    pass
                if self.running:
                    self.root.after(100, poll_queue)
            self.root.after(100, poll_queue)
             # Start the tkinter mainloop
            self.root.mainloop()
            
        except Exception as e:
            print(f"Overlay error: {e}")
    
    def update(self, content: str):
        """Update the overlay content safely"""
        if self.root and self.text_widget and self.running:
            try:
                # Truncate if too long
                if len(content) > 2000:
                    content = content[:2000] + "\n\n[Content truncated...]"
                    
                self.content = content
                
                # Schedule update in tkinter thread
                def do_update():
                    try:
                        self.text_widget.delete('1.0', tk.END)
                        self.text_widget.insert('1.0', content)
                        self.text_widget.see('1.0')  # Scroll to top
                    except:
                        pass
                
                if self.root:
                    self.root.after(0, do_update)
            except Exception as e:
                print(f"Update error: {e}")
        
    def stop(self):
        """Stop the overlay"""
        self.running = False
        if self.root:
            try:
                self.root.after(0, self.root.destroy)
            except:
                pass
