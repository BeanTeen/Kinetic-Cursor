import pyautogui

class actioner:
    def __init__(self):
        pyautogui.FAILSAFE = False
        self.screen_w, self.screen_h = pyautogui.size()

        self.action_map = {
            "move_mouse": self._move,
            "hover_mouse": self._move,
            "mouse_down": lambda p: pyautogui.mouseDown(_pause=False),
            "mouse_up": lambda p: pyautogui.mouseUp(_pause=False),
            "right_click": lambda p: pyautogui.rightClick(_pause=False),
            "copy": lambda p: pyautogui.hotkey('ctrl', 'c', _pause=False),
            "paste": lambda p: pyautogui.hotkey('ctrl', 'v', _pause=False),
            "page_next": lambda p: pyautogui.press("pagedown", _pause=False),
            "page_prev": lambda p: pyautogui.press("pageup", _pause=False),
            "scroll": self._scroll
        }

    def execute(self, actions):
        for event, payload in actions:
            event_func = self.action_map.get(event)
            if event_func:
                event_func(payload)

    def _move(self, payload):
        cx = int(payload[0] * self.screen_w)
        cy = int(payload[1] * self.screen_h)
        pyautogui.moveTo(cx, cy, _pause=False)

    def _scroll(self, payload):
        amount = int(payload['intensity'] * 50)
        if payload['direction'] == 'down':
            amount = -amount
        pyautogui.scroll(amount, _pause=False)