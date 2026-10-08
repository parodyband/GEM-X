"""GEM-X Live for Maya: stream, retarget and record markerless motion capture."""

__version__ = "1.0.0"


def show():
    """Open the GEM-X Live window."""
    from . import ui

    return ui.show()
