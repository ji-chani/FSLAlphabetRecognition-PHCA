# updated to accomodate revamp of mediapipe
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import Literal

VisionRunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = frozenset([
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
])
class Image2Landmarks:
    """ Converts images to landmarks"""
    def __init__(self, model_path:str, landmark_type:Literal["normalized", "world"], flatten:bool=True, 
                 display_image:bool=False, display_landmarks:bool=False, save_figs:bool=False):
        self.model_path = model_path
        self.landmark_type = landmark_type  # hand_landmarks or hand_world_landmarks
        self.flatten = flatten
        self.display_image = display_image
        self.display_landmarks = display_landmarks
        self.save_figs = save_figs

        # build the landmarker once and reuse across calls
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=1,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def __del__(self):
        if hasattr(self, 'landmarker'):
            self.landmarker.close()
    
    def image_to_hand_landmarks(self, image_path:str, figure_size:tuple=(20,10)):
        bgr_image = cv2.imread(image_path)
        rgb_image = _to_mp_image(bgr_image)
    
        results = self.landmarker.detect(rgb_image)
        hand_landmarks = extract_landmarks(results, self.flatten)
        
        # for displaying images and landmarks
        annotated_image = draw_hand_landmarks(bgr_image, results)
        
        if hand_landmarks is not None:
            connected_landmarks = connect_landmarks(hand_landmarks)

            if self.display_image and self.display_landmarks:  # side-by-side annotated iamge and 3d landmarks
                fig = plt.figure(figsize=figure_size)
                ax1 = fig.add_subplot(1,2,1)
                ax1.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
                ax1.axis('off')

                ax2 = fig.add_subplot(1,2,2, projection='3d')
                for points in connected_landmarks:
                    x, y, z = points[0], points[1], points[2]
                    ax2.plot(x,y,z, color='goldenrod', alpha=1)
                    ax2.scatter(x,y,z, color='k')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.tight_layout()

                if self.save_figs:
                    plt.savefig("figures/hand_landmarks.png")
                plt.show()
            
            elif self.display_image and not self.display_landmarks:  # annotated image only
                plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()

            elif not self.display_image and self.display_landmarks:  # 3d landmarks only
                fig = plt.figure(figsize=figure_size)
                ax = fig.add_subplot(projection='3d')
                for points in connected_landmarks:
                    x, y, z = points[0], points[1], points[2]
                    ax.plot(x,y,z, color='goldenrod', alpha=1)
                    ax.scatter(x,y,z, color='k')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.tight_layout()

            if self.save_figs:
                plt.savefig("figures/hand_landmarks.png")
            plt.show()



def _to_mp_image(bgr_frame:np.ndarray) -> mp.Image:
    """ Convert a BGR OpenCV frame to an RGB Mediapipe Image"""
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

def extract_landmarks(results, landmark_type:Literal["normalized", "world"], flattened:bool=True) -> np.ndarray | list | None:
    if landmark_type == "world":
        hand_landmarks = results.hand_world_landmarks
    if landmark_type == "normalized":
        hand_landmarks = results.hand_landmarks

    # results.hand_landmarks is a list of hands
    if not hand_landmarks:
        return None
    
    if len(hand_landmarks) > 1:
        all_hands = []
        for hand in hand_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand])
            if flattened:
                landmarks = landmarks.flatten()
            all_hands.append(landmarks)
        return all_hands
    else:
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks[0]])
        return landmarks.flatten() if flattened else landmarks
    
def draw_hand_landmarks(bgr_image:np.ndarray, results) -> np.ndarray:
    annotated_image = bgr_image.copy()
    h, w = annotated_image.shape[:2]

    # results.hand_landmarks is a list of detected hands
    # each hand being a list of 21 NormalizedLandmarks objects
    for hand in results.hand_landmarks:
        # draw connections
        for a, b in HAND_CONNECTIONS:
            x1, y1 = int(hand[a].x * w), int(hand[a].y * h)
            x2, y2 = int(hand[b].x * w), int(hand[b].y * h)
            cv2.line(img=annotated_image, pt1=(x1,y1), pt2=(x2,y2),
                     color=(229, 228, 226), thickness=2)
        
        # draw dots
        for lm in hand:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img=annotated_image, center=(cx,cy), 
                       radius=2, color=(54,69,79), thickness=2)
    
    return annotated_image

def connect_landmarks(landmarks:np.ndarray) -> list:
    landmarks = landmarks.reshape(21,3)
    connected_landmarks = []
    for a, b in HAND_CONNECTIONS:
        node1, node2 = landmarks[a], landmarks[b]
        x = (node1[0], node2[0])
        y = (node1[1], node2[1])
        z = (node1[2], node2[2])
        connected_landmarks.append([x,y,z])
    return connected_landmarks

def normalize_landmarks(FSLData: np.ndarray) -> np.ndarray:
    FSLData_normalized = (FSLData.copy()).reshape(-1, 21, 3)
    for idx, data in enumerate(FSLData):
        data = data.reshape(21, 3)
        first_row = np.array(data[0].copy())
        FSLData_normalized[idx] = [np.abs(first_row - np.array(data[i])).tolist() for i in range(len(data))]
    return FSLData_normalized.reshape(-1, 21 * 3)
