"""Theme regression checks using real Tk widgets, without starting the driver."""

import tkinter as tk
from tkinter import ttk

import pytest

from autopilot.ui import theme


@pytest.fixture
def root():
    window = tk.Tk()
    window.withdraw()
    yield window
    window.destroy()


def test_light_theme_preserves_native_defaults_and_widget_sizes(root, monkeypatch):
    style = ttk.Style(root)
    native = "vista" if "vista" in style.theme_names() else "clam"
    style.theme_use(native)
    label = ttk.Label(root, text="Active map:")
    button = ttk.Button(root, text="Follow route")
    label.pack()
    button.pack()
    root.update_idletasks()
    sizes = [(w.winfo_reqwidth(), w.winfo_reqheight()) for w in (label, button)]
    options = ("background", "foreground", "padding")
    defaults = {o: style.lookup("TLabel", o) for o in options}
    monkeypatch.setattr(theme, "windows_dark_mode", lambda: False)
    manager = theme.ThemeManager(root, ttk.Label(root))
    manager.apply()
    root.update_idletasks()
    assert [(w.winfo_reqwidth(), w.winfo_reqheight()) for w in (label, button)] == sizes
    assert {o: style.lookup("TLabel", o) for o in options} == defaults
    assert label.winfo_reqwidth() > 30


def test_dark_light_roundtrip_does_not_mutate_stock_theme(root, monkeypatch):
    style = ttk.Style(root)
    style.theme_use("clam")
    original = style.lookup("TLabel", "background")
    manager = theme.ThemeManager(root, ttk.Label(root))
    monkeypatch.setattr(theme, "windows_dark_mode", lambda: True)
    manager.apply()
    assert style.theme_use() == "wardogs-dark"
    assert style.lookup("TLabel", "foreground") == theme.DARK_FG
    monkeypatch.setattr(theme, "windows_dark_mode", lambda: False)
    manager.apply()
    style.theme_use("clam")
    assert style.lookup("TLabel", "background") == original
    monkeypatch.setattr(theme, "windows_dark_mode", lambda: True)
    manager.apply()
    assert style.lookup("TLabel", "foreground") == theme.DARK_FG
