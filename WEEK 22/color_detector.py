import cv2
import numpy as np

def detect_color(frame):
    # Convert the frame from BGR to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges (in HSV)
    colors = {
        "Red": [(0, 120, 70), (10, 255, 255)],
        "Green": [(36, 100, 100), (86, 255, 255)],
        "Blue": [(94, 80, 2), (126, 255, 255)],
        "Yellow": [(22, 93, 200), (45, 255, 255)],
        "Orange": [(10, 100, 20), (25, 255, 255)],
        "Purple": [(129, 50, 70), (158, 255, 255)],
    }

    detected_colors = []

    # Check each color
    for color_name, (lower, upper) in colors.items():
        lower_bound = np.array(lower, dtype=np.uint8)
        upper_bound = np.array(upper, dtype=np.uint8)

        # Create a mask for the current color
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

        # Check if the color exists in the frame
        if cv2.countNonZero(mask) > 0:
            detected_colors.append(color_name)

    return detected_colors

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect colors in the frame
    colors_in_frame = detect_color(frame)

    # Display the detected colors on the frame
    for i, color in enumerate(colors_in_frame):
        cv2.putText(frame, f"{color}", (10, 30 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow("Color Detection", frame)

    # Exit on pressing 'ESC'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
