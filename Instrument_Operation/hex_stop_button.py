import tkinter as tk
import math

class HexStopButton:
    def __init__(self, canvas, cx, cy, size=60, command=None):
        self.canvas = canvas
        self.command = command
        self.tag = "hex_stop_button"

        # Calculate hexagon points
        points = []
        for i in range(6):
            angle_deg = 60 * i 
            angle_rad = math.radians(angle_deg)
            x = cx + size * math.cos(angle_rad)
            y = cy + size * math.sin(angle_rad)
            points.extend([x, y])

        # Draw hexagon and text
        self.canvas.create_polygon(points, fill='red', outline='black', tags=self.tag)
        self.canvas.create_text(cx, cy, text="STOP", fill="white",
                                font=("Helvetica", 20, "bold"), tags=self.tag)

        # Bind click event
        self.canvas.tag_bind(self.tag, "<Button-1>", self._on_click)

    def _on_click(self, event):
        if self.command:
            self.command()
