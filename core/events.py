"""OpenENVI Global Signal & Event Bus.

Provides a decoupled Qt Signal/Slot mechanism for cross-module communication
between UI panels (views, docks, status bar) and backend processing pipelines.
"""

from PySide6.QtCore import QObject, Signal


class AppEventBus(QObject):
    """Global event bus carrying application-wide signals.

    Decouples UI components from each other and from the core engine.
    """

    # Cursor movement and interaction events (pixel coordinates X, Y)
    pixel_hovered = Signal(int, int)
    pixel_clicked = Signal(int, int)

    # Layer selection and state events
    layer_changed = Signal(str)

    # Language selection event ('en' or 'zh')
    language_changed = Signal(str)

    # Display and stretch enhancement events
    stretch_mode_changed = Signal(str)

    # Profile request event (e.g. coordinates triggered for Z-Profile update)
    profile_requested = Signal(int, int)

    # Status notification event (status message, timeout in ms)
    status_message = Signal(str, int)


# Global singleton instance
event_bus = AppEventBus()
