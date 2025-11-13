import sys, os, json, time, collections
import cv2
import mediapipe as mp
import pyautogui

from PyQt5 import QtWidgets, uic, QtCore, QtGui

# Path to the JSON file where gesture → action mapping is stored
CONFIG_PATH = "pose_command_map.json"

# Available actions for each gesture
DEFAULT_ACTIONS = ["none", "left", "right", "up", "down", "shift"]

# Recognized gesture names
GESTURES = ["both_hands_up", "left_hand_up", "right_hand_up", "idle"]


def load_mapping():
    """
    Load gesture → action mapping from JSON.
    If the file is missing or invalid, return a default mapping.
    """
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Ensure all gestures exist in the mapping
            for g in GESTURES:
                data.setdefault(g, "none")
            return data
        except Exception:
            pass

    # Default mapping when there is no valid file
    return {
        "both_hands_up": "up",
        "left_hand_up": "left",
        "right_hand_up": "right",
        "idle": "none",
    }


def save_mapping(mapping):
    """
    Save gesture → action mapping to JSON.
    """
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def key_from_action_name(name):
    """
    Convert the logical action name to a keyboard key
    understood by pyautogui. Return None if no key.
    """
    return name if name in ("left", "right", "up", "down", "shift") else None


class PoseWorker(QtCore.QThread):
    """
    Background worker thread:
    - Reads frames from the camera
    - Runs MediaPipe Pose
    - Detects gestures
    - Converts gestures to key presses
    - Emits QImage frames and status text for the UI
    """

    frameReady = QtCore.pyqtSignal(QtGui.QImage)
    status = QtCore.pyqtSignal(str, str)

    def __init__(self, mapping, parent=None):
        super().__init__(parent)
        self.mapping = mapping
        self._running = True
        self.current_keys = set()
        self.last_gestures = collections.deque(maxlen=5)

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

    def update_keys(self, target_key):
        """
        Keep only one key pressed at a time:
        - Press the new target key (if any)
        - Release all other keys
        """
        desired = set([target_key]) if target_key else set()

        # Press newly desired keys
        for k in desired - self.current_keys:
            pyautogui.keyDown(k)

        # Release keys that are no longer desired
        for k in self.current_keys - desired:
            pyautogui.keyUp(k)

        self.current_keys = desired

    def detect_gesture(self, landmarks):
        """
        Very simple rule-based gesture detection
        using wrist and shoulder y-coordinates.
        """
        lw = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y
        rw = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].y
        ls = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y
        rs = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y

        both = (lw < ls) and (rw < rs)
        left = (lw < ls) and not both
        right = (rw < rs) and not both

        if both:
            return "both_hands_up"
        if left:
            return "left_hand_up"
        if right:
            return "right_hand_up"
        return "idle"

    def run(self):
        """
        Main loop of the worker:
        - Capture frame
        - Process pose
        - Decide gesture and action
        - Update keyboard
        - Emit image and status
        """
        pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        cap = cv2.VideoCapture(0)
        time.sleep(0.4)  # small delay to let the camera warm up

        try:
            while self._running and cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break

                # Mirror the image horizontally
                frame = cv2.flip(frame, 1)

                # Run pose detection (RGB image)
                rgb_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_input)

                # Gesture detection with simple smoothing
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                    )
                    gesture = self.detect_gesture(
                        results.pose_landmarks.landmark
                    )
                    self.last_gestures.append(gesture)
                else:
                    # No landmarks detected → treat as "idle"
                    self.last_gestures.append("idle")

                # Use a majority vote over recent frames
                gesture = collections.Counter(self.last_gestures).most_common(1)[0][0]

                # Look up action and corresponding keyboard key
                action_name = self.mapping.get(gesture or "idle", "none")
                target_key = key_from_action_name(action_name)

                # Update pressed keys according to current action
                self.update_keys(target_key)

                # Overlay gesture and action text on the frame
                cv2.putText(
                    frame,
                    f"Gesture: {gesture or 'none'}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Action:  {action_name}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                # Convert frame (BGR) to QImage (RGB) for Qt
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                bytes_per_line = ch * w
                qimg = QtGui.QImage(
                    rgb.data,
                    w,
                    h,
                    bytes_per_line,
                    QtGui.QImage.Format_RGB888,
                ).copy()

                # Emit signals to the UI
                self.frameReady.emit(qimg)
                self.status.emit(gesture or "none", action_name)

        finally:
            # Make sure we release all keys and the camera
            self.update_keys(None)
            try:
                cap.release()
            except Exception:
                pass
            cv2.destroyAllWindows()

    def stop(self):
        """
        Stop the worker loop.
        """
        self._running = False


class SettingsWindow(QtWidgets.QMainWindow):
    """
    Window based on pose_setting.ui.
    This window lets the user choose the mapping for each gesture.
    When the user presses "Start", it emits mappingChosen(mapping).
    """

    mappingChosen = QtCore.pyqtSignal(dict)
    closed = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        uic.loadUi("pose_setting.ui", self)

        # Widgets in pose_setting.ui
        self.cb_both = self.findChild(
            QtWidgets.QComboBox, "comboBox_Both_hands_up"
        )
        self.cb_left = self.findChild(
            QtWidgets.QComboBox, "comboBox_Left_hand_up"
        )
        self.cb_right = self.findChild(
            QtWidgets.QComboBox, "comboBox_Right_hand_up"
        )
        self.cb_idle = self.findChild(
            QtWidgets.QComboBox, "comboBox_Idle"
        )

        self.btn_start = self.findChild(QtWidgets.QPushButton, "StartButton")
        self.btn_close = self.findChild(QtWidgets.QPushButton, "CloseButton")

        # Fill combo boxes with action choices
        for cb in (self.cb_both, self.cb_left, self.cb_right, self.cb_idle):
            cb.addItems(DEFAULT_ACTIONS)

        # Load previously saved mapping and apply to combo boxes
        mapping = load_mapping()
        self.set_mapping(mapping)

        # Connect buttons
        self.btn_start.clicked.connect(self.on_start_clicked)
        self.btn_close.clicked.connect(self.close)

    def set_mapping(self, mapping: dict):
        """
        Set combo box values based on a mapping dictionary.
        """
        self.cb_both.setCurrentText(mapping.get("both_hands_up", "none"))
        self.cb_left.setCurrentText(mapping.get("left_hand_up", "none"))
        self.cb_right.setCurrentText(mapping.get("right_hand_up", "none"))
        self.cb_idle.setCurrentText(mapping.get("idle", "none"))

    def read_mapping_from_ui(self):
        """
        Read current mapping values from the combo boxes.
        """
        return {
            "both_hands_up": self.cb_both.currentText(),
            "left_hand_up": self.cb_left.currentText(),
            "right_hand_up": self.cb_right.currentText(),
            "idle": self.cb_idle.currentText(),
        }

    def on_start_clicked(self):
        """
        Called when the Start button is pressed:
        - Read current mapping from combo boxes
        - Save mapping to JSON file
        - Emit mappingChosen(mapping) to the controller
        """
        mapping = self.read_mapping_from_ui()
        save_mapping(mapping)
        self.mappingChosen.emit(mapping)

    def closeEvent(self, e):
        """
        When the settings window is closed, emit 'closed'
        so the controller can exit the app if needed.
        """
        self.closed.emit()
        return super().closeEvent(e)


class MainWindow(QtWidgets.QMainWindow):
    """
    Window based on pose.ui.
    This window shows the camera image and recognized gesture/action.
    It runs the PoseWorker thread and has a "Setting" button
    to go back to the settings window.
    """

    requestSettings = QtCore.pyqtSignal()

    def __init__(self, mapping: dict):
        super().__init__()
        uic.loadUi("pose.ui", self)

        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)

        # Widgets in pose.ui
        self.video_label = self.findChild(QtWidgets.QLabel, "videoLabel")
        self.gesture_label = self.findChild(QtWidgets.QLabel, "gestureLabel")
        self.action_label = self.findChild(QtWidgets.QLabel, "actionLabel")

        self.btn_setting = self.findChild(
            QtWidgets.QPushButton, "Setting_Button"
        )
        self.btn_close = self.findChild(
            QtWidgets.QPushButton, "Close_Button"
        )

        self.btn_setting.clicked.connect(self.on_setting_clicked)
        self.btn_close.clicked.connect(self.close)

        self.mapping = mapping
        self.worker = None
        self.last_frame = None

        # Start pose detection and camera capture immediately
        self.start_worker()

    def start_worker(self):
        """
        Start the PoseWorker thread if it is not already running.
        """
        if self.worker is not None:
            return
        self.worker = PoseWorker(self.mapping)
        self.worker.frameReady.connect(self.on_frame)
        self.worker.status.connect(self.on_status)
        self.worker.start()

    def stop_worker(self):
        """
        Safely stop the PoseWorker thread.
        """
        if self.worker:
            self.worker.stop()
            self.worker.wait(1000)
            self.worker = None

    def update_mapping_and_restart(self, mapping: dict):
        """
        Called when a new mapping is chosen from the settings window.
        Stop the worker, update mapping, and restart detection.
        """
        self.mapping = mapping
        self.stop_worker()
        self.start_worker()

    @QtCore.pyqtSlot()
    def on_setting_clicked(self):
        """
        Called when the Setting button is pressed.
        Ask the controller to show the settings window.
        """
        self.requestSettings.emit()

    @QtCore.pyqtSlot(QtGui.QImage)
    def on_frame(self, qimg):
        """
        Receive a new frame from the PoseWorker and draw it on the label.
        """
        self.last_frame = qimg
        pix = QtGui.QPixmap.fromImage(qimg)
        pix = pix.scaled(
            self.video_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(pix)

    def resizeEvent(self, e):
        """
        When the window is resized, redraw the last frame
        so that it fits the new label size.
        """
        if self.last_frame is not None:
            self.on_frame(self.last_frame)
        return super().resizeEvent(e)

    @QtCore.pyqtSlot(str, str)
    def on_status(self, gesture, action):
        """
        Update text labels for current gesture and action.
        """
        self.gesture_label.setText(f"Gesture: {gesture}")
        self.action_label.setText(f"Action: {action}")

    def closeEvent(self, e):
        """
        When the main window is closed, stop the worker.
        """
        self.stop_worker()
        return super().closeEvent(e)


class AppController(QtCore.QObject):
    """
    Controller that manages:
    - Showing the settings window first
    - Creating / showing the main window (pose.ui)
    - Switching between them when "Setting" is pressed
    """

    def __init__(self):
        super().__init__()
        self.settings = SettingsWindow()
        self.main = None

        self.settings.mappingChosen.connect(self.on_mapping_chosen)
        self.settings.closed.connect(self.on_settings_closed)

        # Show settings window when the application starts
        self.settings.show()

    @QtCore.pyqtSlot(dict)
    def on_mapping_chosen(self, mapping: dict):
        """
        Called when the user presses Start in the settings window.
        If this is the first time, create the main window.
        Otherwise, just update the mapping.
        Then show the main window and hide the settings window.
        """
        if self.main is None:
            self.main = MainWindow(mapping)
            self.main.requestSettings.connect(self.on_request_settings)
        else:
            self.main.update_mapping_and_restart(mapping)

        self.main.show()
        self.settings.hide()

    @QtCore.pyqtSlot()
    def on_request_settings(self):
        """
        Called when the main window's Setting button is pressed.
        Stop the worker, hide the main window, and show the settings window.
        """
        if self.main:
            self.main.stop_worker()
            self.main.hide()

        # Sync current mapping back to the settings combos
        if self.main:
            self.settings.set_mapping(self.main.mapping)

        self.settings.show()

    @QtCore.pyqtSlot()
    def on_settings_closed(self):
        """
        If the settings window is closed (without main window),
        quit the entire application.
        """
        QtWidgets.QApplication.quit()


def main():
    """
    Entry point of the application.
    """
    app = QtWidgets.QApplication(sys.argv)
    controller = AppController()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
