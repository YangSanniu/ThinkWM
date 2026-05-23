"""Mock psychopy for testing — avoids display/GUI dependencies."""
import sys, types, numpy as np

class MockPsychopy:
    """Mock out psychopy modules so thinkWM can be imported without a display."""

    @staticmethod
    def install():
        mock_visual = types.ModuleType('psychopy.visual')
        mock_core = types.ModuleType('psychopy.core')
        mock_event = types.ModuleType('psychopy.event')
        mock_gui = types.ModuleType('psychopy.gui')
        mock_tools = types.ModuleType('psychopy.tools')
        mock_monitors = types.ModuleType('psychopy.monitors')

        # Mock classes
        class MockWindow:
            def __init__(self, *a, **kw): pass
            def flip(self): pass
            def close(self): pass
        class MockRect:
            def __init__(self, *a, **kw): self.pos = (0, 0); self.fillColor = None
            def draw(self): pass
        class MockTextStim:
            def __init__(self, *a, **kw): pass
            def draw(self): pass
        class MockClock:
            def __init__(self): self._time = 0.0
            def getTime(self): return self._time
            def reset(self, t=0.0): self._time = t
        class MockMouse:
            def __init__(self, *a, **kw): pass
            def setPos(self, *a): pass
            def setVisible(self, *a): pass
            def getPressed(self): return [False]
            def isPressedIn(self, *a): return False

        mock_visual.Window = MockWindow
        mock_visual.Rect = MockRect
        mock_visual.TextStim = MockTextStim
        mock_core.Clock = MockClock
        mock_core.wait = lambda t: None
        mock_core.quit = lambda: None
        mock_event.Mouse = MockMouse
        mock_event.waitKeys = lambda *a, **kw: None
        mock_event.getKeys = lambda *a, **kw: []
        mock_gui.DlgFromDict = lambda *a, **kw: type('dlg', (), {'OK': False})
        mock_tools.coordinatetools = types.ModuleType('coordinatetools')
        mock_tools.coordinatetools.pol2cart = lambda a, r: (0.0, 0.0)
        mock_monitors.Monitor = lambda *a, **kw: type('m', (), {'setSizePix': lambda s: None})()

        # Also mock the parent modules
        sys.modules['psychopy'] = types.ModuleType('psychopy')
        sys.modules['psychopy.visual'] = mock_visual
        sys.modules['psychopy.core'] = mock_core
        sys.modules['psychopy.event'] = mock_event
        sys.modules['psychopy.gui'] = mock_gui
        sys.modules['psychopy.tools'] = mock_tools
        sys.modules['psychopy.tools.coordinatetools'] = mock_tools.coordinatetools
        sys.modules['psychopy.monitors'] = mock_monitors
