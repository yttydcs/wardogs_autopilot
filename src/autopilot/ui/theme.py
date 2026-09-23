"""Studio visual theme: follows the Windows dark/light mode.

The palette constants live here so both the App widgets and any canvas
helpers reference a single source of truth.
"""

import tkinter as tk
from tkinter import ttk

DARK_BG = "#242424"
DARK_PANEL = "#2d2d2d"
DARK_HI = "#3f3f3f"
DARK_FG = "#e8e8e8"
DARK_CANVAS = "#2b2b2b"
RGB_CANVAS = (43, 43, 43)  # equivalent of #2b2b2b for the numpy background

def windows_dark_mode() -> bool:
    """Windows app dark theme (AppsUseLightTheme == 0)."""
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
        ) as k:
            return winreg.QueryValueEx(k, "AppsUseLightTheme")[0] == 0
    except (ImportError, OSError):
        return False


class ThemeManager:
    """Applies (and live-tracks) the dark/light palette against the OS theme.

    root       — the Tk window being themed.
    roi_status — the "Capture zone" status label whose color flips by theme.
    """

    def __init__(self, root: tk.Tk, roi_status) -> None:
        self._root = root
        self._roi_status = roi_status
        self._style = ttk.Style(root)
        self._dark = False

    def apply(self) -> None:
        """Synchronize the widget palette with the current Windows theme."""
        dark = windows_dark_mode()
        self._dark = dark
        s = self._style
        if not dark:
            s.theme_use("vista" if "vista" in s.theme_names() else "clam")
            # Native themes already provide valid defaults. Empty style values
            # override those defaults and can collapse widgets or render black.
            self._root.option_add("*TCombobox*Listbox.background", "white")
            self._root.option_add("*TCombobox*Listbox.foreground", "black")
            self._root.option_add("*TCombobox*Listbox.selectBackground", "#0078d7")
            self._root.option_add("*TCombobox*Listbox.selectForeground", "white")
            self._root.configure(bg="#f0f0f0")
            self._roi_status.config(foreground="green")
            return
        # Keep dark overrides out of the stock themes used in light mode.
        if "wardogs-dark" not in s.theme_names():
            s.theme_create("wardogs-dark", parent="clam")
        s.theme_use("wardogs-dark")
        bg, fg, panel, hi = DARK_BG, DARK_FG, DARK_PANEL, DARK_HI
        s.configure(
            ".",
            background=bg,
            foreground=fg,
            troughcolor="#1a1a1a",
            bordercolor=hi,
            darkcolor=bg,
            lightcolor=hi,
            focuscolor="#1b6ac9",
            selectbackground="#1b6ac9",
            selectforeground=fg,
        )
        s.configure("TFrame", background=bg)
        s.configure("TLabel", background=bg, foreground=fg)
        s.configure("TNotebook", background=bg, borderwidth=0)
        s.configure(
            "TNotebook.Tab", background=panel, foreground=fg, padding=(10, 4), borderwidth=0
        )
        s.map("TNotebook.Tab", background=[("selected", hi)], foreground=[("selected", "#ffffff")])
        s.configure(
            "TButton", background=hi, foreground=fg, bordercolor=hi, padding=(8, 3), focuscolor=hi
        )
        s.map(
            "TButton",
            background=[("active", "#4e4e4e"), ("pressed", "#333333")],
            foreground=[("active", fg)],
        )
        s.configure("TEntry", fieldbackground=panel, foreground=fg, bordercolor=hi, insertcolor=fg)
        s.configure(
            "TSpinbox", fieldbackground=panel, foreground=fg, bordercolor=hi, insertcolor=fg
        )
        s.configure(
            "TCombobox",
            fieldbackground=panel,
            foreground=fg,
            bordercolor=hi,
            background=hi,
            arrowcolor=fg,
            insertcolor=fg,
        )
        s.map(
            "TCombobox",
            fieldbackground=[("readonly", panel)],
            foreground=[("readonly", fg)],
            background=[("active", "#4e4e4e"), ("pressed", "#333333")],
            arrowcolor=[("disabled", "#6f6f6f")],
        )
        self._root.option_add("*TCombobox*Listbox.background", panel)
        self._root.option_add("*TCombobox*Listbox.foreground", fg)
        self._root.option_add("*TCombobox*Listbox.selectBackground", "#1b6ac9")
        self._root.option_add("*TCombobox*Listbox.selectForeground", fg)
        s.configure("Horizontal.TScale", background=bg, troughcolor="#151515")
        s.configure(
            "Vertical.TScrollbar", background=hi, troughcolor=bg, bordercolor=bg, arrowcolor=fg
        )
        s.configure(
            "Horizontal.TScrollbar", background=hi, troughcolor=bg, bordercolor=bg, arrowcolor=fg
        )
        self._root.configure(bg=bg)
        self._roi_status.config(foreground="#8ae234")

    def start_poll(self) -> None:
        """Begin polling the OS theme every 2 s and re-applying on change."""
        self._root.after(2000, self._poll)

    def _poll(self) -> None:
        try:
            if windows_dark_mode() != self._dark:
                self.apply()
        finally:
            self._root.after(2000, self._poll)
