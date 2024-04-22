from mediapipe import solutions
import cv2
import numpy as np

class ClassifyHand:
    """
    Finds hand(s) using the MediaPipe library.
    Exports the landmarks (either in pixel format or not).
    Provides bounding box for the hands.
    Classify the signed letter using a specified classifier.
    """

    def __init__(self, scaler, classifier, train_dim, static_mode=False, max_hands=2, model_complexity=1, detection_conf=0.5, tracking_conf=0.5):
        """
        :param scaler: trained feature scaler for preprocessing
        :param classifier: trained classifier for recognition
        :param train_dim: dimension of images in training set of classifier
        :param static_mode: in static mode, detection is done on each image: slower
        "param: max_hands: maximum number of hands to detect
        :param: model_complexity: complexity of the hand landmark model: 0 or 1
        :param: detection_conf: minimum detection confidence threshold
        :param: min_tracking_conf: minimum tracking confidence threshold
        """

        self.scaler = scaler
        self.classifier = classifier
        self.train_dim = train_dim
        self.static_mode = static_mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_conf = detection_conf
        self.tracking_conf = tracking_conf
        self.mp_hands = solutions.hands
        self.mp_drawing = solutions.drawing_utils
        self.hands = self.mp_hands.Hands(static_image_mode=self.static_mode,
                                         max_num_hands=self.max_hands,
                                         model_complexity=self.model_complexity,
                                         min_detection_confidence=self.detection_conf,
                                         min_tracking_confidence=self.tracking_conf)
        

    def classify_hand_to_letter(self, img, draw=True, flip_type=True):
        """
        detect hands from image, extract landmarks, and predict signed letter
        :param img: image or frame to find the hands from
        :param draw: flag to draw the output on the image or frame
        :return: Image with or without the interpreted letter
        """

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        train_h, train_w = self.train_dim[0], self.train_dim[1]  # dimension of training images
        height, width, _ = img_rgb.shape
        self.results = self.hands.process(img_rgb)  # landmark detection using MediaPipe
        all_hands = []
        if self.results.multi_hand_landmarks:
            for landmarks in self.results.multi_hand_landmarks:
                my_hand = {}
                xlist = [int(lm.x * width) for lm in landmarks.landmark]
                ylist = [int(lm.y * height) for lm in landmarks.landmark]

                # bbox (in pixels)
                xmin, xmax = min(xlist), max(xlist)
                ymin, ymax = min(ylist), max(ylist)
                bbox_width, bbox_height = xmax-xmin, ymax-ymin
                cx, cy = xmin + (bbox_width // 2), ymin + (bbox_height // 2)
                my_hand['center'] = (cx, cy)
                my_hand['bbox'] = [xmin, ymin, bbox_width, bbox_height]

                # updating bbox based in training set
                new_xmin, new_xmax = cx - (train_w // 2), cx + (train_w // 2)
                new_ymin, new_ymax = cy - (train_h // 2), cy + (train_h // 2)
                new_img = img_rgb[new_xmin:new_xmax, new_ymin:new_ymax, :]
                self.new_results = self.hands.process(new_img)

                # landmark recognition
                if self.new_results.multi_hand_landmarks:
                    for lmarks in self.new_results.multi_hand_landmarks:
                        landmark_list = [[lmark.x, lmark.y, lmark.z] for lmark in lmarks.landmark]
                        my_hand['label'] = predict_landmark(landmark_list, self.scaler, self.classifier)
                all_hands.append(my_hand)

                # drawing
                if draw:
                    self.mp_drawing.draw_landmarks(img, landmarks, self.mp_hands.HAND_CONNECTIONS)
                    cv2.rectangle(img,
                                  (new_xmin, new_ymin),
                                  (new_xmax, new_ymax),
                                  (255, 0, 255), 2)
                    cv2.putText(img, my_hand['label'],
                                (new_xmin-30, new_ymin-30),
                                cv2.FONT_HERSHEY_PLAIN,
                                2, (255,0,255), 2)
        return all_hands, img
    
def predict_landmark(landmark_list, scaler, classifier):
    """ 
    scale and classify landmarks (scale -> classify)
    :param landmark_list: List of landmarks of hand with shape (21,3)
    :scaler: Standard Scaler
    :classifier: Machine Learning classifier
    :return: Classified letter
    """
    landmark_list = np.array(landmark_list).reshape(1,63)
    test_data = scaler.transform(landmark_list)
    prediction = classifier.predict(test_data)
    return prediction[0]