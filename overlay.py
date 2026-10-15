import tkinter as tk
import threading

class overlay:

    def __init__(self):
        self.x = 0
        self.y = 0
        self.thread = threading.Thread(target=self._run_tk, daemon=True)
        self.thread.start()

    def update_pos(self, x, y):
        self.x = x
        self.y = y

    def _run_tk(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.wm_attributes('-topmost', True)
        self.root.wm_attributes('-disabled', True)
        self.root.wm_attributes('-transparentcolor', 'black')

        self.size = 50
        self.canvas = tk.Canvas(self.root, width=self.size, height=self.size, bg='black', highlightthickness=0)
        self.canvas.pack()

        self.canvas.create_oval(2, 2, self.size-2, self.size-2, outline='cyan', width=3)
        self.canvas.create_oval(self.size//2 - 2, self.size//2 - 2, self.size//2 - 2, self.size//2 - 2, fill='cyan')

        self._update_loop()
        self.root.mainloop()

    def _update_loop(self):
        offset = self.size//2
        self.root.geometry(f"+{self.x - offset}+{self.y - offset}")

        self.root.after(16, self._update_loop)