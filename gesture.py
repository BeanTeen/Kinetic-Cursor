import math
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import config

class gesture:
    #thin wrapper around HandLandmarker
    def __init__(self, model_path=config.MODEL_PATH, num_hands=config.NUM_HANDS): 
        base_options = python.BaseOptions(model_asset_path = model_path) #configure landmarker
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                            num_hands=num_hands,
                                            running_mode=vision.RunningMode.VIDEO
                                            )
        
        self._landmarker = vision.HandLandmarker.create_from_options(options) #create landmarker module
        self._timeStamp = 0 #track runtime in milliseconds

    def process(self, frame): #put frame through model to look for hand
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self._timeStamp += 33 #shows timestamp in ms

        res = self._landmarker.detect_for_video(mp_image, self._timeStamp)
        if res and res.hand_landmarks:
            return res.hand_landmarks[0]
        return None

    def close(self):
        if self._landmarker:
            self._landmarker.close()

class gestureClassifier:

    @staticmethod
    def classify(landmarks) -> dict:
        return {
            'index_pinch_dist': gestureClassifier._dist(landmarks[4], landmarks[8]),
            'middle_pinch_dist': gestureClassifier._dist(landmarks[4], landmarks[12]),
            'is_fist': gestureClassifier._is_fist(landmarks),
            'is_open_palm': gestureClassifier._is_open_palm(landmarks),
            'pointing_dir': gestureClassifier._pointing_direction(landmarks),
            'cursor_pos': (landmarks[8].x, landmarks[8].y)
        }

    @staticmethod
    def _dist(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    @staticmethod
    def _finger_extended(landmarks, mcp_idx, tip_idx):
        wrist = landmarks[0]
        mcp = landmarks[mcp_idx]
        tip = landmarks[tip_idx]
        return gestureClassifier._dist(tip, wrist) > config.EXTENSION_RATIO * gestureClassifier._dist(mcp, wrist)

    @staticmethod
    def _is_fist(landmarks):
        fingers = [(5, 8), (9, 12), (13, 16), (17, 20)]
        return not any(gestureClassifier._finger_extended(landmarks, m, t) for m, t in fingers)

    @staticmethod
    def _is_open_palm(landmarks):
        fingers = [(5, 8), (9, 12), (13, 16), (17, 20)]
        return all(gestureClassifier._finger_extended(landmarks, m, t) for m, t in fingers)

    @staticmethod
    def _pointing_direction(landmarks):
        index_ext = gestureClassifier._finger_extended(landmarks, 5, 8)
        others_curled = not any(gestureClassifier._finger_extended(landmarks, m, t) for m, t in[(9, 12), (13, 16), (17, 20)])

        if index_ext and others_curled:
            dy = landmarks[8].y - landmarks[5].y
            dx = landmarks[8].x - landmarks[5].x

            dy_safe = abs(dy) if abs(dy) > 0.0001 else 0.0001
            if (abs(dx) / dy_safe) > config.POINT_HORIZONTAL_RATIO:
                if dx < -0.05:
                    return 'left'
                elif dx > 0.05:
                    return 'right'
        return None
        
class cursorSmoother:
    #Exponential moving average on cursor position

    def __init__(self, alpha=config.SMOOTHING_ALPHA):
        self.alpha = alpha
        self._smoothed = None

    def update(self, pos):
        if self._smoothed is None:
            self._smoothed = pos
        else:
            x = self._smoothed[0] + self.alpha * (pos[0] - self._smoothed[0])
            y = self._smoothed[1] + self.alpha * (pos[1] - self._smoothed[1])
            self._smoothed = (x, y)
        return self._smoothed

    def reset(self):
        self._smoothed = None

class gestureStateMachine:
    #Stateful: tracks gesture history across frames and terms them into clean events

    def __init__(self,
        db_frames=config.DEBOUNCE_FRAMES,
        cooldown_ms=config.DISCRETE_COOLDOWN_MS,
        scroll_rep_ms=config.SCROLL_REPEAT_MS
    ):
        self.db_frames = db_frames
        self.cooldown_ms = cooldown_ms
        self.scroll_rep_ms = scroll_rep_ms

        self._index_pinch_frames = 0
        self._middle_pinch_frames = 0
        self._fist_frames = 0
        self._palm_frames = 0

        self._is_dragging = False
        self._middle_pinch_armed = True
        self._fist_armed = True
        self._palm_armed = True
        self._last_fist_time = 0
        self._last_palm_time = 0

        self._last_scroll_time = 0

    def update(self, signals: dict, timestamp_ms: int) -> list:
        events = []
        pos = signals.get('cursor_pos')

        events.extend(self._update_index_pinch(signals, pos))
        events.extend(self._update_middle_pinch(signals, pos))
        events.extend(self._update_fist(signals, timestamp_ms))
        events.extend(self._update_palm(signals, timestamp_ms))
        events.extend(self._update_scroll(signals, timestamp_ms))

        return events

    def _update_index_pinch(self, signals, cursor_pos):
        events = []
        
        dist = signals['index_pinch_dist']
        if dist < config.PINCH_THRESHOLD:
            self._index_pinch_frames += 1
            if self._index_pinch_frames >=self.db_frames and not self._is_dragging:
                self._is_dragging = True
                events.append(('mouse_down', cursor_pos))
            elif self._is_dragging:
                events.append(('mouse_move', cursor_pos))
        elif dist > config.PINCH_RELEASE_THRESHOLD:
            if self._is_dragging:
                events.append(('mouse_up', cursor_pos))
                self._is_dragging = False
            self._index_pinch_frames = 0
            if not self._is_dragging:
                events.append(('hover_move', cursor_pos))
        else:#maintin state in deadzone
            if self._is_dragging:
                events.append(('mouse_move', cursor_pos))
            else:
                self._index_pinch_frames = 0
                events.append(('hover_move', cursor_pos))
        return events

    def _update_middle_pinch(self, signals, cursor_pos):
        events = []
        
        dist = signals['middle_pinch_dist']

        if dist < config.PINCH_THRESHOLD:
            if self._middle_pinch_armed:
                self._middle_pinch_frames += 1
                if self._middle_pinch_frames >= self.db_frames:
                    events.append(('right_click', cursor_pos))
                    self._middle_pinch_armed = False
        elif dist > config.PINCH_RELEASE_THRESHOLD:
            self._middle_pinch_armed = True
            self._middle_pinch_frames = 0

        return events

    def _update_fist(self, signals, timestamp_ms):
        events = []

        time_since_last = timestamp_ms - self._last_fist_time
        is_cooled = time_since_last >= self.cooldown_ms

        if signals['is_fist']:
            if self._fist_armed and is_cooled:
                self._fist_frames += 1
                if self._fist_frames >= self.db_frames:
                    events.append(('copy', None))
                    self._fist_armed = False
                    self._last_fist_time = timestamp_ms
        else:
            self._fist_armed = True
            self._fist_frames = 0

        return events

    def _update_palm(self, signals, timestamp_ms):
        events = []

        time_since_last = timestamp_ms - self._last_palm_time
        is_cooled = time_since_last >= self.cooldown_ms

        if signals['is_open_palm']:
            if self._palm_armed and is_cooled:
                self._palm_frames += 1
                if self._palm_frames >= self.db_frames:
                    events.append(('paste', None))
                    self._palm_armed = False
                    self._last_palm_time = timestamp_ms
        else:
            self._palm_armed = True
            self._palm_frames = 0

        return events

    def _update_scroll(self, signals, timestamp_ms):
        events = []
        direction = signals['pointing_dir']
        if direction:
            if (timestamp_ms - self._last_scroll_time) >= self.scroll_rep_ms:
                events.append(('scroll', direction))
                self._last_scroll_time = timestamp_ms
        else:
            self._last_scroll_time = 0
        return events
    
if __name__ == "__main__":
    class MockConfig:
        DEBOUNCE_FRAMES = 2
        DISCRETE_COOLDOWN_MS = 500
        SCROLL_REPEAT_MS = 100
        PINCH_THRESHOLD = 0.04
        PINCH_RELEASE_THRESHOLD = 0.04
    
    config = MockConfig()
    gsm = gestureStateMachine(db_frames=2)
    
    mock_sequence = [
        {'index_pinch_dist': 0.05, 'middle_pinch_dist': 0.1, 'is_fist': False, 'is_open_palm': False, 'pointing_dir': None, 'cursor_pos': (0.5, 0.5)}, 
        {'index_pinch_dist': 0.02, 'middle_pinch_dist': 0.1, 'is_fist': False, 'is_open_palm': False, 'pointing_dir': None, 'cursor_pos': (0.5, 0.5)}, 
        {'index_pinch_dist': 0.01, 'middle_pinch_dist': 0.1, 'is_fist': False, 'is_open_palm': False, 'pointing_dir': None, 'cursor_pos': (0.6, 0.5)}, 
        {'index_pinch_dist': 0.01, 'middle_pinch_dist': 0.1, 'is_fist': False, 'is_open_palm': False, 'pointing_dir': None, 'cursor_pos': (0.7, 0.5)}, 
        {'index_pinch_dist': 0.06, 'middle_pinch_dist': 0.1, 'is_fist': False, 'is_open_palm': False, 'pointing_dir': None, 'cursor_pos': (0.7, 0.5)}, 
    ]
    
    for i, signals in enumerate(mock_sequence):
        print(f"Frame {i}:", gsm.update(signals, i * 33))