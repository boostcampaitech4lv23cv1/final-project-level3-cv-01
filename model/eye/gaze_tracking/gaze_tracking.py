from __future__ import division
import os
import cv2
import dlib
from .eye import Eye
from .calibration import Calibration
import pandas as pd


class GazeTracking(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()
        self.VIDEO_PATH = ""
        self.SAVED_DIR = ""

        # _face_detector is used to detect faces
        self._face_detector = dlib.get_frontal_face_detector()

        # _predictor is used to get facial landmarks of a given face
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(
            os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat")
        )
        self._predictor = dlib.shape_predictor(model_path)

    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def _analyze(self):
        """Detects the face and initialize Eye objects"""
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(frame)

        try:
            landmarks = self._predictor(frame, faces[0])
            self.eye_left = Eye(frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(frame, landmarks, 1, self.calibration)

        except IndexError:
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)
        else:
            return (0, 0)

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)
        else:
            return (0, 0)

    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def is_right(self):
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.38

    def is_left(self):
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.62

    def is_up(self):
        if self.pupils_located:
            return self.vertical_ratio() >= 0.62

    def is_down(self):
        if self.pupils_located:
            return self.vertical_ratio() <= 0.38

    def is_center(self):
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            return (
                self.is_right() is not True
                and self.is_left() is not True
                and self.is_up() is not True
                and self.is_down() is not True
            )

    def is_blinking(self):
        """Returns true if the user closes his eyes"""
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 3.8

    def annotated_frame(self):
        """Returns the main frame with pupils highlighted"""
        frame = self.frame.copy()

        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

        return frame

    def analyze_eye(self, frames):
        """analyze eye tracking used in fastapi"""
        ret = []
        left = []
        right = []
        anno_frames = []

        for frame in frames:

            # self.annotated_frame()
            if self.is_right() or self.is_left() or self.is_up() or self.is_down():
                text = "Side"
            elif self.is_center():
                text = "Center"
            else:
                text = "None"

            # if None, return (0,0)
            left_pupil = self.pupil_left_coords()
            right_pupil = self.pupil_right_coords()

            ret.append(text)
            left.append(left_pupil)
            right.append(right_pupil)

            # annotation
            anno_frame = self.get_annotated_frame(frame, text, left_pupil, right_pupil)
            anno_frames.append(anno_frame)

        df = pd.DataFrame({"tracking": ret, "left": left, "right": right})

        return df, anno_frames

    def get_annotated_frame(self, path, text, left_pupil, right_pupil):
        frame = cv2.imread(path)
        self.refresh(frame)
        frame = self.annotated_frame()

        if text == "None":
            cv2.putText(
                frame, text, (90, 60), cv2.FONT_HERSHEY_TRIPLEX, 1.6, (0, 0, 255), 3
            )
            cv2.putText(
                frame,
                "Left:  " + str(left_pupil),
                (90, 130),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.9,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                "Right: " + str(right_pupil),
                (90, 165),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.9,
                (0, 0, 255),
                2,
            )

        else:
            cv2.putText(
                frame, text, (90, 60), cv2.FONT_HERSHEY_TRIPLEX, 1.6, (0, 255, 0), 3
            )
            cv2.putText(
                frame,
                "Left:  " + str(left_pupil),
                (90, 130),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "Right: " + str(right_pupil),
                (90, 165),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

        return frame

    def frame_to_video(self, rec_image_list):
        cap = cv2.VideoCapture(self.VIDEO_PATH)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"vp80")

        out = cv2.VideoWriter(f"./db/eye.webm", fourcc, 2, (width, height))

        for rec_frame in rec_image_list:
            out.write(rec_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
