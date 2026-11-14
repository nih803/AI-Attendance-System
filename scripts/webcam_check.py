import cv2

# Open the default webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open the webcam.")
else:
    print("Webcam started! Press Q to quit.")

    while True:
        # Read one frame from the webcam
        success, frame = camera.read()

        if not success:
            print("Could not read the webcam frame.")
            break

        # Display the live video
        cv2.imshow("AI Attendance System - Webcam", frame)

        # Quit when the user presses Q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release the webcam and close windows
camera.release()
cv2.destroyAllWindows()
