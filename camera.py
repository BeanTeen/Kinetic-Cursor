import cv2
import sys
import config
from overlay import overlay
from gesture import gesture as ges, gestureClassifier as gc

POINTS = {"wrist": 0, "thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
COLOR_1 = (0, 255, 0)
COLOR_2 = (255, 0, 0)
def main():
    cam = cv2.VideoCapture(config.CAM_INDEX)
    frame_w = config.FRAME_WIDTH
    frame_h = config.FRAME_HEIGHT
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)

    if not cam.isOpened():
        print("Could Not Open Camera")
        sys.exit(1)

    tracker = ges()

    while True:
        ret, frame = cam.read()

        if not ret:
            print("Could Not Connect To Camera")
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        handPoints = tracker.process(rgb_frame)

        if handPoints:
            for n, (name, index) in enumerate(POINTS.items()):
                point = handPoints[index]
                px, py = int(point.x * frame_w), int(point.y * frame_h)
                cv2.circle(frame, (px, py), radius=5, color=COLOR_1, thickness=-1)
                cv2.putText(frame, name, (px+12, py-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_2, 2)
                cv2.putText(frame, f"{name}: ({px}, {py})  ({point.x:.3f}, {point.y:.3f})", (10, 65+n*25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_2, 2)
            thumb_idx = handPoints[POINTS["thumb"]]
            index_idx = handPoints[POINTS["index"]]

            t_x = int(thumb_idx.x * frame_w)
            t_y = int(thumb_idx.y * frame_h)
            i_x = int(index_idx.x * frame_w)
            i_y = int(index_idx.y * frame_h)

            mid_x = int(t_x + i_x) // 2
            mid_y = int(t_y + i_y) // 2

            cv2.line(frame, (t_x, t_y), (i_x, i_y), (0, 255, 255), 2)
            cv2.circle(frame, (mid_x, mid_y), radius=8, color=(0, 255, 255), thickness=-1)

            signals = gc.classify(handPoints)

            if signals['is_fist']:
                label = "Fist"
            elif signals['is_open_palm']:
                label = "Open Palm"
            elif signals['pointing_dir']:
                label = f"Pointing {signals['pointing_dir']}"
            elif signals['is_middle_finger']:
                label = "Fuck you, too"
            elif signals['index_pinch_dist'] < config.PINCH_THRESHOLD:
                label = "Pinch (Index)"
            elif signals['middle_pinch_dist'] < config.PINCH_THRESHOLD:
                label = "Pinch (Middle)"
            else:
                label = "No Gesture Detected"

            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_1, 2)

        cv2.imshow("Point Tracker", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    tracker.close()

if __name__ == "__main__":
    main()
