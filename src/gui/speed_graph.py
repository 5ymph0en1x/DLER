"""
DLER Speed Graph Widget
=======================

Windows 11-style animated speed graph using pure Tkinter Canvas.
Shows download speed history with gradient fill area chart.
"""

from __future__ import annotations

import tkinter as tk
from collections import deque
from typing import Optional, Tuple
import math


class SpeedGraphWidget(tk.Canvas):
    """
    Animated speed graph widget with Windows 11-style gradient fill.

    Features:
    - Gradient area chart with glow effect
    - Auto-scaling Y-axis based on max speed
    - Rolling 60-second window of data
    - EMA smoothing for stable display
    - Grid lines and current speed label
    """

    # Colors matching DLER dark theme (park dark)
    # Park dark theme background is ~#2b2b2b, we use slightly darker
    BG_COLOR = "#252525"          # Close to interface background
    GRID_COLOR = "#353535"        # Subtle grid lines
    LINE_COLOR = "#217346"        # Forest green (park theme accent)
    GLOW_COLOR = "#217346"        # Matching glow
    TEXT_COLOR = "#909090"        # Muted text
    ACCENT_COLOR = "#2d9254"      # Lighter forest green for highlights

    # Area fill colors (forest green tones for gradient effect)
    FILL_COLOR_DARK = "#1a2a1a"   # Dark green-gray for bottom
    FILL_COLOR_MID = "#1f3a25"    # Mid forest green
    FILL_COLOR_LIGHT = "#254a2d"  # Lighter green near line

    def __init__(
        self,
        parent,
        width: int = 400,
        height: int = 100,
        max_samples: int = 60,
        update_interval_ms: int = 100,
        **kwargs
    ):
        """
        Initialize the speed graph widget.

        Args:
            parent: Parent Tkinter widget
            width: Canvas width in pixels
            height: Canvas height in pixels
            max_samples: Number of samples to display (default 60 = 60 seconds at 1/s)
            update_interval_ms: Animation update interval
        """
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=self.BG_COLOR,
            highlightthickness=0,
            **kwargs
        )

        self._width = width
        self._height = height
        self._max_samples = max_samples
        self._update_interval = update_interval_ms

        # Data storage with EMA smoothing
        self._speed_history: deque[float] = deque(maxlen=max_samples)
        self._current_speed: float = 0.0
        self._ema_speed: float = 0.0
        self._ema_alpha: float = 0.3  # Smoothing factor

        # Auto-scaling Y-axis
        self._max_speed_seen: float = 10.0  # Start with 10 MB/s minimum
        self._scale_decay: float = 0.995  # Slow decay for max scale

        # Drawing margins
        self._margin_left = 45
        self._margin_right = 10
        self._margin_top = 8
        self._margin_bottom = 20

        # Pre-compute graph area
        self._graph_width = width - self._margin_left - self._margin_right
        self._graph_height = height - self._margin_top - self._margin_bottom

        # Gradient cache (more strips = smoother gradient)
        self._gradient_colors = self._generate_gradient(30)

        # Animation state
        self._animation_id: Optional[str] = None
        self._is_running: bool = False

        # Initialize with zeros
        for _ in range(max_samples):
            self._speed_history.append(0.0)

        # Draw initial state
        self._draw_frame()

    def _generate_gradient(self, steps: int) -> list[str]:
        """Generate gradient colors from bottom (dark) to top (lighter green)."""
        colors = []

        # Three-point gradient: dark -> mid -> light
        r1, g1, b1 = self._hex_to_rgb(self.FILL_COLOR_DARK)
        r2, g2, b2 = self._hex_to_rgb(self.FILL_COLOR_MID)
        r3, g3, b3 = self._hex_to_rgb(self.FILL_COLOR_LIGHT)

        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0

            # Two-phase interpolation for richer gradient
            if t < 0.5:
                # Dark to mid
                t2 = t * 2
                r = int(r1 + (r2 - r1) * t2)
                g = int(g1 + (g2 - g1) * t2)
                b = int(b1 + (b2 - b1) * t2)
            else:
                # Mid to light
                t2 = (t - 0.5) * 2
                r = int(r2 + (r3 - r2) * t2)
                g = int(g2 + (g3 - g2) * t2)
                b = int(b2 + (b3 - b2) * t2)

            colors.append(f"#{r:02x}{g:02x}{b:02x}")

        return colors

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def update_speed(self, speed_mbps: float) -> None:
        """
        Update the graph with a new speed value.

        Args:
            speed_mbps: Current download speed in MB/s
        """
        self._current_speed = max(0.0, speed_mbps)

        # Apply EMA smoothing
        self._ema_speed = (
            self._ema_alpha * self._current_speed +
            (1 - self._ema_alpha) * self._ema_speed
        )

        # Add to history
        self._speed_history.append(self._ema_speed)

        # Update max scale (with slow decay for smooth scaling)
        if self._ema_speed > self._max_speed_seen:
            self._max_speed_seen = self._ema_speed * 1.2  # 20% headroom
        else:
            # Slowly decay max to allow scale to shrink over time
            self._max_speed_seen = max(
                10.0,  # Minimum scale
                self._max_speed_seen * self._scale_decay,
                max(self._speed_history) * 1.2 if self._speed_history else 10.0
            )

    def start_animation(self) -> None:
        """Start the animation loop."""
        if not self._is_running:
            self._is_running = True
            self._animate()

    def stop_animation(self) -> None:
        """Stop the animation loop."""
        self._is_running = False
        if self._animation_id:
            self.after_cancel(self._animation_id)
            self._animation_id = None

    def reset(self) -> None:
        """Reset the graph to initial state."""
        self._speed_history.clear()
        for _ in range(self._max_samples):
            self._speed_history.append(0.0)
        self._current_speed = 0.0
        self._ema_speed = 0.0
        self._max_speed_seen = 10.0
        self._draw_frame()

    def _animate(self) -> None:
        """Animation loop."""
        if not self._is_running:
            return

        self._draw_frame()
        self._animation_id = self.after(self._update_interval, self._animate)

    def _draw_frame(self) -> None:
        """Draw a single frame of the graph."""
        self.delete("all")

        # Draw background grid
        self._draw_grid()

        # Draw the area chart with gradient
        self._draw_area_chart()

        # Draw the line on top
        self._draw_speed_line()

        # Draw Y-axis labels
        self._draw_y_labels()

        # Draw current speed value
        self._draw_current_speed()

    def _draw_grid(self) -> None:
        """Draw background grid lines."""
        x0 = self._margin_left
        x1 = self._width - self._margin_right
        y0 = self._margin_top
        y1 = self._height - self._margin_bottom

        # Horizontal grid lines (4 lines)
        for i in range(5):
            y = y0 + (y1 - y0) * i / 4
            self.create_line(
                x0, y, x1, y,
                fill=self.GRID_COLOR,
                width=1,
                dash=(2, 4)
            )

        # Vertical grid lines (6 lines for 60 seconds)
        for i in range(7):
            x = x0 + (x1 - x0) * i / 6
            self.create_line(
                x, y0, x, y1,
                fill=self.GRID_COLOR,
                width=1,
                dash=(2, 4)
            )

        # Border
        self.create_rectangle(
            x0, y0, x1, y1,
            outline=self.GRID_COLOR,
            width=1
        )

    def _draw_area_chart(self) -> None:
        """Draw gradient-filled area under the speed line."""
        if not self._speed_history or self._max_speed_seen <= 0:
            return

        x0 = self._margin_left
        x1 = self._margin_left + self._graph_width
        y_bottom = self._height - self._margin_bottom
        y_top = self._margin_top

        samples = list(self._speed_history)
        n_samples = len(samples)

        if n_samples < 2:
            return

        # Pre-compute data points (x, y) for the curve
        data_points = []
        for i, speed in enumerate(samples):
            x = x0 + (self._graph_width * i / (n_samples - 1))
            y = self._speed_to_y(speed)
            data_points.append((x, y))

        # Draw gradient using horizontal strips from bottom to top
        n_strips = len(self._gradient_colors)
        strip_height = (y_bottom - y_top) / n_strips

        for strip_idx, color in enumerate(self._gradient_colors):
            # Strip boundaries (bottom strip = index 0, top strip = last index)
            strip_y_bottom = y_bottom - strip_idx * strip_height
            strip_y_top = strip_y_bottom - strip_height

            # Build polygon for this strip: bottom-left -> curve points -> bottom-right
            strip_points = []

            # Start at bottom-left of strip
            strip_points.append(x0)
            strip_points.append(strip_y_bottom)

            # Add curve points, clamped to strip boundaries
            for px, py in data_points:
                # Clamp Y to be within the strip (top is smaller Y value)
                clamped_y = max(strip_y_top, min(strip_y_bottom, py))
                strip_points.append(px)
                strip_points.append(clamped_y)

            # End at bottom-right of strip
            strip_points.append(x1)
            strip_points.append(strip_y_bottom)

            # Draw the strip polygon
            if len(strip_points) >= 6:  # At least 3 points
                self.create_polygon(
                    strip_points,
                    fill=color,
                    outline="",
                    width=0
                )

    def _draw_speed_line(self) -> None:
        """Draw the speed line on top of the area with antialiasing effect."""
        if not self._speed_history:
            return

        samples = list(self._speed_history)
        n_samples = len(samples)

        if n_samples < 2:
            return

        x0 = self._margin_left

        # Build line points
        points = []
        for i, speed in enumerate(samples):
            x = x0 + (self._graph_width * i / (n_samples - 1))
            y = self._speed_to_y(speed)
            points.extend([x, y])

        # Antialiasing: draw glow layers underneath (darker to lighter)
        # Layer 1: Dark glow (wider)
        self.create_line(
            points,
            fill=self.FILL_COLOR_MID,  # Dark green glow
            width=5,
            smooth=True,
            splinesteps=24,
            capstyle="round",
            joinstyle="round"
        )

        # Layer 2: Medium glow
        self.create_line(
            points,
            fill=self.FILL_COLOR_LIGHT,  # Medium green
            width=3,
            smooth=True,
            splinesteps=24,
            capstyle="round",
            joinstyle="round"
        )

        # Main line (on top)
        self.create_line(
            points,
            fill=self.LINE_COLOR,
            width=2,
            smooth=True,
            splinesteps=24,  # Higher = smoother curve
            capstyle="round",
            joinstyle="round"
        )

        # Draw highlight dot at current position
        if points:
            last_x, last_y = points[-2], points[-1]
            # Outer glow
            self.create_oval(
                last_x - 5, last_y - 5,
                last_x + 5, last_y + 5,
                fill="",
                outline=self.GLOW_COLOR,
                width=2
            )
            # Inner dot
            self.create_oval(
                last_x - 3, last_y - 3,
                last_x + 3, last_y + 3,
                fill=self.ACCENT_COLOR,
                outline=""
            )

    def _draw_y_labels(self) -> None:
        """Draw Y-axis scale labels."""
        x = self._margin_left - 3
        y0 = self._margin_top
        y1 = self._height - self._margin_bottom

        # Draw 3 labels: max, mid, 0 (compact format)
        max_label = self._format_speed_compact(self._max_speed_seen)
        mid_label = self._format_speed_compact(self._max_speed_seen / 2)

        self.create_text(
            x, y0,
            text=max_label,
            anchor="e",
            fill=self.TEXT_COLOR,
            font=("Consolas", 7)
        )

        self.create_text(
            x, (y0 + y1) / 2,
            text=mid_label,
            anchor="e",
            fill=self.TEXT_COLOR,
            font=("Consolas", 7)
        )

        self.create_text(
            x, y1,
            text="0",
            anchor="e",
            fill=self.TEXT_COLOR,
            font=("Consolas", 7)
        )

    def _draw_current_speed(self) -> None:
        """Draw current speed value in top-right corner."""
        speed_text = self._format_speed(self._ema_speed)

        x = self._width - self._margin_right - 5
        y = self._margin_top + 12

        # Background for readability
        self.create_text(
            x, y,
            text=speed_text,
            anchor="e",
            fill=self.ACCENT_COLOR,
            font=("Consolas", 11, "bold")
        )

    def _speed_to_y(self, speed: float) -> float:
        """Convert speed value to Y coordinate."""
        if self._max_speed_seen <= 0:
            return self._height - self._margin_bottom

        ratio = min(1.0, speed / self._max_speed_seen)
        y0 = self._margin_top
        y1 = self._height - self._margin_bottom

        return y1 - ratio * (y1 - y0)

    @staticmethod
    def _format_speed(speed: float) -> str:
        """Format speed for display."""
        if speed < 1.0:
            return f"{speed * 1024:.0f} KB/s"
        elif speed < 100:
            return f"{speed:.1f} MB/s"
        elif speed < 1000:
            return f"{speed:.0f} MB/s"
        else:
            return f"{speed / 1024:.2f} GB/s"

    @staticmethod
    def _format_speed_compact(speed: float) -> str:
        """Format speed in compact form for Y-axis labels."""
        if speed < 1.0:
            return f"{speed * 1024:.0f}K"
        elif speed < 10:
            return f"{speed:.1f}M"
        elif speed < 1000:
            return f"{speed:.0f}M"
        else:
            return f"{speed / 1024:.1f}G"


# Demo / test code
if __name__ == "__main__":
    import random

    root = tk.Tk()
    root.title("DLER Speed Graph Demo")
    root.configure(bg="#2b2b2b")  # Park dark theme background
    root.geometry("560x140")

    # Create widget (same size as in DLER GUI)
    graph = SpeedGraphWidget(root, width=520, height=70)
    graph.pack(padx=10, pady=10)

    # Simulate speed updates with realistic pattern
    speed = 0.0
    ramp_up = True

    def update():
        global speed, ramp_up
        if ramp_up:
            # Ramp up to ~80 MB/s
            speed = min(80, speed + random.uniform(2, 8))
            if speed >= 75:
                ramp_up = False
        else:
            # Fluctuate around current speed
            speed = max(0, speed + random.uniform(-8, 8))
            speed = min(120, speed)

        graph.update_speed(speed)
        root.after(100, update)

    graph.start_animation()
    root.after(500, update)  # Start after brief delay

    root.mainloop()
