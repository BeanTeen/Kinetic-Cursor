import cv2
import sys
import config
from gesture import gesture as ges

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
            for point in handPoints:
                px = int(point.x * frame_w)
                py = int(point.y * frame_h)

                cv2.circle(frame, (px, py), radius=5, color=(0,255,0), thickness=-1)
        cv2.imshow("Point Tracker", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    tracker.close()

if __name__ == "__main__":
    main()
