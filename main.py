# Import the OpenCV library for video processing and display
import cv2

# Import the YOLO class from the ultralytics package for object detection
from ultralytics import YOLO

# Load the pre-trained custom YOLOv8 smoke detection model from the local file
# NOTE: rename your trained weights file to 'smoke.pt' (or update the path below)
model = YOLO('smoke.pt')

# Open the input video file for reading frames
cap = cv2.VideoCapture('input_video.mp4')

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open input_video.mp4")
    exit()

# Get the width of the video frames (in pixels)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Get the height of the video frames (in pixels)
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Get the frames per second (FPS) of the input video
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec to use for the output video (mp4v works well for .mp4 files)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Create a VideoWriter object to save the output video
# It uses the same resolution and FPS as the input video
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))

# Define the color for the bounding box and text (Orange/Amber in BGR format)
# Amber is used instead of red to visually distinguish "early warning" (smoke)
# from "active danger" (fire) alerts
SMOKE_COLOR = (0, 165, 255)

# Start an infinite loop to process the video frame by frame
while True:

    # Read one frame from the video
    # 'ret' is True if the frame was read successfully, 'frame' contains the image data
    ret, frame = cap.read()

    # If no frame was returned, it means the video has ended, so break the loop
    if not ret:
        print("End of video reached or cannot read the frame.")
        break

    # Run the YOLOv8 model on the current frame to detect smoke
    # 'results' contains all the detection information (boxes, confidence, etc.)
    results = model(frame)

    # Loop through each detection result (usually just one per frame)
    for result in results:

        # Loop through each bounding box detected in this result
        for box in result.boxes:

            # Extract the coordinates of the bounding box (top-left and bottom-right corners)
            x1, y1, x2, y2 = box.xyxy[0]

            # Convert the coordinates from float to integer for drawing
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Get the confidence score of this detection
            confidence = float(box.conf[0])

            # Draw a bounding box around the detected smoke
            cv2.rectangle(frame, (x1, y1), (x2, y2), SMOKE_COLOR, 2)

            # Create the label text to display, including the confidence score
            label = f"SMOKE DETECTED ({confidence:.2f})"

            # Overlay the "SMOKE DETECTED" text above the bounding box
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, SMOKE_COLOR, 2)

    # Write the processed frame (with any drawn boxes/labels) into the output video file
    out.write(frame)

    # Display the processed frame in a window titled "Smoke Detection"
    cv2.imshow("Smoke Detection", frame)

    # Wait 1 millisecond for a key press; if the key is 'q', stop the loop early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object to free up the camera/video file resource
cap.release()

# Release the video writer object to finalize and save the output video file
out.release()

# Close all OpenCV display windows
cv2.destroyAllWindows()

# Print a confirmation message once processing is complete
print("Processing complete. Output saved as output_video.mp4")
