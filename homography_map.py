import cv2
import numpy as np

def select_homography_points(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image:", image_path)
        return None, None

    points = []

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

    cv2.imshow("Select 4 ground points (clockwise)", image)
    cv2.setMouseCallback("Select 4 ground points (clockwise)", click)

    while len(points) < 4:
        cv2.imshow("Select 4 ground points (clockwise)", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    if len(points) != 4:
        print("Not enough points selected.")
        return None, None

    src_pts = np.array(points, dtype=np.float32)
    width, height = 500, 500
    dst_pts = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src_pts, dst_pts)
    return H, (width, height)

def transform_point(H, point):
    pt = np.array([[point]], dtype=np.float32)
    dst = cv2.perspectiveTransform(pt, H)
    x, y = dst[0][0]
    return int(x), int(y)