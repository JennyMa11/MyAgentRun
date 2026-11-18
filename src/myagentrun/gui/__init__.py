"""Optional local browser presentation, independent of the core runtime."""

from myagentrun.gui.server import handle_connection, serve_gui
from myagentrun.gui.session import GuiSession

__all__ = ["GuiSession", "handle_connection", "serve_gui"]
