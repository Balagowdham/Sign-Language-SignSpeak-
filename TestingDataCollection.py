# Webcam capture tool for building the TESTING dataset.
# Press a letter key (a-z) to save the current hand ROI into
# dataSet/testingData/<LETTER>/, press '0' for the blank class, 'Esc' to quit.
#
# Same output structure as the original data-collection script; refactored
# from 26 near-duplicate if-blocks into a single lookup-driven loop.

import os
import string

import cv2

MODE = "testingData"
DIRECTORY = os.path.join("dataSet", MODE)
THRESHOLD_MIN = 70
ROI_DISPLAY_SIZE = 300


def counts_for(directory):
    labels = ["0"] + list(string.ascii_uppercase)
    return {label: len(os.listdir(os.path.join(directory, label))) for label in labels}


def draw_counts(frame, counts):
    y = 60
    for label in ["0"] + list(string.ascii_uppercase):
        cv2.putText(
            frame, f"{label} : {counts[label]}", (10, y),
            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1,
        )
        y += 10


def main():
    capture = cv2.VideoCapture(0)

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        counts = counts_for(DIRECTORY)
        draw_counts(frame, counts)

        # Region of interest for the hand
        x1 = int(0.5 * frame.shape[1])
        y1 = 10
        x2 = frame.shape[1] - 10
        y2 = int(0.5 * frame.shape[1])
        cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
        roi = frame[y1:y2, x1:x2]
        cv2.imshow("Frame", frame)

        # Image processing (grayscale -> blur -> adaptive threshold -> otsu)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 2)
        adaptive = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        _, processed = cv2.threshold(
            adaptive, THRESHOLD_MIN, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        preview = cv2.resize(processed, (ROI_DISPLAY_SIZE, ROI_DISPLAY_SIZE))
        cv2.imshow("Processed ROI", preview)

        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # Esc
            break

        label = None
        if key == ord("0"):
            label = "0"
        elif ord("a") <= key <= ord("z"):
            label = chr(key).upper()

        if label is not None:
            out_path = os.path.join(DIRECTORY, label, f"{counts[label]}.jpg")
            cv2.imwrite(out_path, roi)

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
