import cv2
import numpy as np
from scipy.spatial import distance as dist
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from collections import OrderedDict

# ROI points as a numpy array of coordinates for RGB images
roi_points = np.array([
    [850, 535],  # bottom right corner
    [100, 535],  # bottom-left corner
    [200, 435],  # top-left corner
    [750, 435]   # top-right corner
], dtype=np.int32)

# ROI points as a numpy array of coordinates for Thermal images
# roi_points = np.array([
#     [620, 510],  # bottom right corner
#     [20, 510],  # bottom-left corner
#     [70, 415],  # top-left corner
#     [570, 415]   # top-right corner
# ], dtype=np.int32)



# Create a subclass of the Config class
class PredictionConfig(Config):
    NAME = "cattle_cfg_coco"
    NUM_CLASSES = 1 + 1  # Background + cow
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    USE_MINI_MASK = False
    DETECTION_MIN_CONFIDENCE = 0.6

# Setup configuration and model
cfg = PredictionConfig()
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
model.load_weights('logs/cattle_cfg_coco20231212T2340/mask_rcnn_cattle_cfg_coco_0025.h5', by_name=True)

# Video file to process
video_path = 'drone video/DJI_20230316142010_0003_S_low.mp4'  # RGB
#video_path = 'drone video/DJI_20230316142010_0003_T.mp4'  #Thermal


# Function to draw the ROI polygon on the image
def draw_roi(image, roi_points, color=(0, 0, 255), thickness=3):
    roi_points = roi_points.reshape((-1, 1, 2))
    return cv2.polylines(image, [roi_points], isClosed=True, color=color, thickness=thickness)


# Function to check if a point is inside a given polygon
def is_inside_polygon(roi, point):
    # Ensure point is a tuple of floats which OpenCV expects
    point = (float(point[0]), float(point[1]))
    return cv2.pointPolygonTest(roi, point, False) > 0

# Centroid Tracker Class
class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects

# Instantiate the tracker
tracker = CentroidTracker()

total_cows_counted = 0  # Initialize the total cow count
highest_id_ever_assigned = 0

# Open the video file
capture = cv2.VideoCapture(video_path)

# Define the codec and create VideoWriter object to save the output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video_RGB.avi', fourcc, 20.0, (int(capture.get(3)), int(capture.get(4))))


# Process the video
while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    # Convert the frame to the format expected by the model (if necessary)
    frame_rgb = frame[:, :, ::-1]

    # Perform detection
    detected = model.detect([frame_rgb])
    r = detected[0]

    # Initialize a list for storing detections
    rects = []

    # Filter detections by confidence and ROI
    for i in range(r['rois'].shape[0]):
        if r['scores'][i] > cfg.DETECTION_MIN_CONFIDENCE:
            y1, x1, y2, x2 = r['rois'][i]
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            # Check if the centroid is within the ROI
            if is_inside_polygon(roi_points, centroid):
                rects.append((x1, y1, x2, y2))

    # Update the tracker with the filtered detections
    objects = tracker.update(rects)
    
    # Check if we have a new highest ID, update the cumulative count if so
    current_max_id = max(objects.keys(), default=-1)
    highest_id_ever_assigned = max(highest_id_ever_assigned, current_max_id)

    # Draw the ROI polygon
    cv2.polylines(frame, [roi_points], True, (0, 0, 255), 2)

    # Draw the detected cows and their IDs
    for (objectID, centroid) in objects.items():
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # Display the total count of unique cows detected
    #total_cows_counted = len(objects)
    #cv2.putText(frame, "Total Cows: {}".format(total_cows_counted),(frame.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    
    # Display the cumulative count of unique cows detected
    cv2.putText(frame, "Total Cows: {}".format(highest_id_ever_assigned + 1), (frame.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    
    # Write the frame
    out.write(frame)
    
    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the 'q' key was pressed, break from the loop
    if key == ord('q'):
        break

# Cleanup
capture.release()
out.release()
cv2.destroyAllWindows()