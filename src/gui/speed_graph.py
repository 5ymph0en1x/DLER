"""
DLER Speed Graph Widget
=======================

Elegant speed graph showing complete download history.
All data points are preserved and compressed to fit the display.
Resets on each new transfer.
"""

from __future__ import annotations

import tkinter as tk
from typing import Optional, Tuple, List
import math


class SpeedGraphWidget(tk.Canvas):
    """
    Speed graph widget showing complete transfer history.

    Features:
    - Fine anti-aliased line with subtle glow
    - Preserves all data points (compresses X-axis)
    - Auto-scaling Y-axis
    - Gradient fill under curve
    - Resets on each new transfer
    """

    # Colors matching DLER dark theme (park dark)
    BG_COLOR = "#2b2b2b"          # Match interface background
    GRID_COLOR = "#383838"        # Subtle grid lines
    LINE_COLOR = "#217346"        # Forest green
    LINE_GLOW = "#1a5c38"         # Darker green for glow
    TEXT_COLOR = "#808080"
    ACCENT_COLOR = "#2d9254"      # Lighter green for current speed

    # Gradient fill colors (subtle, blending with interface)
    FILL_COLORS = [
        "#1f251f",  # Bottom - dark with hint of green
        "#212721",
        "#232923",
        "#252b25",
        "#273027",
        "#293529",  # Top - subtle green
    ]

    def __init__(
        self,
        parent,
        width: int = 400,
        height: int = 100,
        update_interval_ms: int = 100,
        **kwargs
    ):
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
        self._update_interval = update_interval_ms

        # Data storage - list grows indefinitely, no sliding window
        self._speed_history: List[float] = []
        self._current_speed: float = 0.0
        self._ema_speed: float = 0.0
        self._ema_alpha: float = 0.25

        # Auto-scaling Y-axis
        self._max_speed_seen: float = 10.0
        self._display_max: float = 10.0  # Smoothed display max

        # Haute couture margins - maximized graph area
        self._margin_left = 26    # Just enough for "999M" labels
        self._margin_right = 50   # Compact speed indicator space
        self._margin_top = 2      # Minimal top breathing room
        self._margin_bottom = 8   # Reduced bottom spacing

        # Pre-compute graph area
        self._graph_x0 = self._margin_left
        self._graph_x1 = width - self._margin_right
        self._graph_y0 = self._margin_top
        self._graph_y1 = height - self._margin_bottom
        self._graph_width = self._graph_x1 - self._graph_x0
        self._graph_height = self._graph_y1 - self._graph_y0

        # Animation state
        self._animation_id: Optional[str] = None
        self._is_running: bool = False

        # Draw initial empty state
        self._draw_frame()

    def update_speed(self, speed_mbps: float) -> None:
        """Add a new speed sample."""
        self._current_speed = max(0.0, speed_mbps)

        # EMA smoothing
        if not self._speed_history:
            self._ema_speed = self._current_speed
        else:
            self._ema_speed = (
                self._ema_alpha * self._current_speed +
                (1 - self._ema_alpha) * self._ema_speed
            )

        self._speed_history.append(self._ema_speed)

        # Update max scale
        if self._ema_speed > self._max_speed_seen:
            self._max_speed_seen = self._ema_speed

        # Smooth display max (gradual scaling)
        target_max = max(10.0, self._max_speed_seen * 1.15)
        self._display_max += (target_max - self._display_max) * 0.1

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
        """Reset for a new transfer."""
        self._speed_history.clear()
        self._current_speed = 0.0
        self._ema_speed = 0.0
        self._max_speed_seen = 10.0
        self._display_max = 10.0
        self._draw_frame()

    def _animate(self) -> None:
        """Animation loop."""
        if not self._is_running:
            return
        self._draw_frame()
        self._animation_id = self.after(self._update_interval, self._animate)

    def _draw_frame(self) -> None:
        """Draw a complete frame."""
        self.delete("all")
        self._draw_grid()

        if len(self._speed_history) >= 2:
            points = self._compute_display_points()
            self._draw_gradient_fill(points)
            self._draw_line(points)
            self._draw_endpoint(points)

        self._draw_labels()

    def _compute_display_points(self) -> List[Tuple[float, float]]:
        """Compute display points, downsampling if needed for performance."""
        samples = self._speed_history
        n = len(samples)

        if n == 0:
            return []

        # Maximum points to render (for performance)
        max_points = min(n, self._graph_width)

        points = []

        if n <= max_points:
            # Use all points
            for i, speed in enumerate(samples):
                x = self._graph_x0 + (self._graph_width * i / max(1, n - 1))
                y = self._speed_to_y(speed)
                points.append((x, y))
        else:
            # Downsample using LTTB-like algorithm (largest triangle)
            step = n / max_points
            for i in range(max_points):
                idx = int(i * step)
                # Average nearby samples for smoother downsampling
                start_idx = max(0, idx - 1)
                end_idx = min(n, idx + 2)
                avg_speed = sum(samples[start_idx:end_idx]) / (end_idx - start_idx)

                x = self._graph_x0 + (self._graph_width * i / max(1, max_points - 1))
                y = self._speed_to_y(avg_speed)
                points.append((x, y))

        return points

    def _draw_grid(self) -> None:
        """Draw subtle grid."""
        x0, x1 = self._graph_x0, self._graph_x1
        y0, y1 = self._graph_y0, self._graph_y1

        # Horizontal lines (3 divisions)
        for i in range(4):
            y = y0 + (y1 - y0) * i / 3
            self.create_line(x0, y, x1, y, fill=self.GRID_COLOR, width=1)

        # Border
        self.create_rectangle(x0, y0, x1, y1, outline=self.GRID_COLOR, width=1)

    def _draw_gradient_fill(self, points: List[Tuple[float, float]]) -> None:
        """Draw gradient fill under the curve."""
        if len(points) < 2:
            return

        n_bands = len(self.FILL_COLORS)
        band_height = self._graph_height / n_bands

        for band_idx, color in enumerate(self.FILL_COLORS):
            band_bottom = self._graph_y1 - band_idx * band_height
            band_top = band_bottom - band_height

            # Build polygon for this band
            poly = [self._graph_x0, band_bottom]

            for px, py in points:
                clamped_y = max(band_top, min(band_bottom, py))
                poly.extend([px, clamped_y])

            poly.extend([self._graph_x1, band_bottom])

            if len(poly) >= 6:
                self.create_polygon(poly, fill=color, outline="")

    def _draw_line(self, points: List[Tuple[float, float]]) -> None:
        """Draw anti-aliased line with subtle glow."""
        if len(points) < 2:
            return

        # Flatten points for create_line
        flat = []
        for x, y in points:
            flat.extend([x, y])

        # Single fine line (no glow)
        self.create_line(
            flat,
            fill=self.LINE_COLOR,
            width=1,
            smooth=True,
            splinesteps=64,
            capstyle="round",
            joinstyle="round"
        )

    def _draw_endpoint(self, points: List[Tuple[float, float]]) -> None:
        """Draw small indicator at current position."""
        if not points:
            return

        x, y = points[-1]
        r = 2  # Tiny dot

        # Simple dot (no glow)
        self.create_oval(
            x - r, y - r, x + r, y + r,
            fill=self.LINE_COLOR, outline=""
        )

    def _draw_labels(self) -> None:
        """Draw Y-axis labels and current speed."""
        # Y-axis labels (max and 0)
        max_label = self._format_compact(self._display_max)

        self.create_text(
            self._graph_x0 - 2, self._graph_y0 + 1,
            text=max_label, anchor="e",
            fill=self.TEXT_COLOR, font=("Consolas", 7)
        )
        self.create_text(
            self._graph_x0 - 2, self._graph_y1 - 1,
            text="0", anchor="e",
            fill=self.TEXT_COLOR, font=("Consolas", 7)
        )

        # Current speed (right side, vertically centered in margin)
        speed_text = self._format_speed(self._ema_speed)
        self.create_text(
            self._width - 3, self._graph_y0 + 6,
            text=speed_text, anchor="e",
            fill=self.ACCENT_COLOR, font=("Consolas", 9, "bold")
        )

    def _speed_to_y(self, speed: float) -> float:
        """Convert speed to Y coordinate."""
        if self._display_max <= 0:
            return self._graph_y1
        ratio = min(1.0, speed / self._display_max)
        return self._graph_y1 - ratio * self._graph_height

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
    def _format_compact(speed: float) -> str:
        """Compact format for Y-axis."""
        if speed < 1.0:
            return f"{speed * 1024:.0f}K"
        elif speed < 10:
            return f"{speed:.1f}M"
        elif speed < 1000:
            return f"{speed:.0f}M"
        else:
            return f"{speed / 1024:.1f}G"


# Demo
if __name__ == "__main__":
    import random

    root = tk.Tk()
    root.title("DLER Speed Graph")
    root.configure(bg="#2b2b2b")
    root.geometry("700x120")

    graph = SpeedGraphWidget(root, width=650, height=70)
    graph.pack(padx=20, pady=20)

    speed = 0.0
    phase = 0
    sample_count = 0

    def update():
        global speed, phase, sample_count
        sample_count += 1

        # Simulate realistic download pattern
        if phase == 0:  # Ramp up
            speed = min(85, speed + random.uniform(3, 10))
            if speed >= 80:
                phase = 1
        elif phase == 1:  # Steady with fluctuation
            speed = max(20, min(100, speed + random.uniform(-5, 5)))
            if sample_count > 200:
                phase = 2
        elif phase == 2:  # Slow down
            speed = max(0, speed - random.uniform(1, 5))
            if speed <= 5:
                phase = 3
        else:  # Done, reset
            graph.reset()
            speed = 0.0
            phase = 0
            sample_count = 0

        graph.update_speed(speed)
        root.after(50, update)

    graph.start_animation()
    root.after(200, update)
    root.mainloop()
