from mediapipe import solutions
import matplotlib.pyplot as plt
import numpy as np
import cv2

# for building keypoints
mp_hands = solutions.hands
mp_drawing = solutions.drawing_utils
mp_drawing_styles = solutions.drawing_styles

class Image2Landmarks:
    def __init__(self, flatten:bool=True, display_image:bool=False):
        self.flatten = flatten
        self.display = display_image
    
    def image_to_hand_landmarks(self, image_path:str):
        image = cv2.imread(image_path)
        # image = cv2.resize(imagem (150, 150))

        with mp_hands.Hands(model_complexity=0.5, max_num_hands=2, min_tracking_confidence=0.5) as hands:
            image, results = mediapipe_detection(image, hands)
            hand_landmarks = extract_landmarks(results, self.flatten)
            if self.display:
                annotated_image = draw_hand_landmarks(image, results)
                plt.imshow(annotated_image)
                plt.axis('off')
        return hand_landmarks


##### HELPER FUNCTION ########
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_hand_landmarks(image, results):
    annotated_image = image.copy()
    if results.multi_hand_landmarks:
        for num, hand in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(annotated_image, hand, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(54, 69, 79), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(229, 228, 226), thickness=2, circle_radius=2))
    return annotated_image

def extract_landmarks(results, flattened:bool):
    hand_landmarks = None  # 21 landmarks, 3 keypoints, 1 hand
    if results.multi_handedness:
        hand_landmarks = np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[0].landmark])
        if flattened:
            hand_landmarks = hand_landmarks.flatten()
    return hand_landmarks

def normalize_landmarks(FSLData):
    FSLData_normalized = (FSLData.copy()).reshape(-1,21,3)
    for idx, data in enumerate(FSLData):
        data = data.reshape(21,3)
        first_row = np.array(data[0].copy())
        FSLData_normalized[idx] = [np.abs(first_row - np.array(data[i])).tolist() for i in range(len(data))]
    return FSLData_normalized.reshape(-1, 21*3)