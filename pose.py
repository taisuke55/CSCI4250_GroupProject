import sys, os, json, time, collections
import cv2
import mediapipe as mp
import pyautogui

from PyQt5 import QtWidgets, uic, QtCore, QtGui

CONFIG_PATH = "pose_command_map.json"
DEFAULT_ACTIONS = ["none", "left", "right", "up", "down", "shift"]
GESTURES = ["both_hands_up", "left_hand_up", "right_hand_up", "idle"]

def load_mapping():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for g in GESTURES:
                data.setdefault(g, "none")
            return data
        except Exception:
            pass
    return {"both_hands_up": "up", "left_hand_up": "left", "right_hand_up": "right", "idle": "none"}

def save_mapping(m):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)

def key_from_action_name(name):
    return name if name in ("left","right","up","down","shift") else None

class PoseWorker(QtCore.QThread):
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
        desired = set([target_key]) if target_key else set()
        for k in desired - self.current_keys:
            pyautogui.keyDown(k)
        for k in self.current_keys - desired:
            pyautogui.keyUp(k)
        self.current_keys = desired

    def detect_gesture(self, lms):
        lw = lms[self.mp_pose.PoseLandmark.LEFT_WRIST].y
        rw = lms[self.mp_pose.PoseLandmark.RIGHT_WRIST].y
        ls = lms[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y
        rs = lms[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        both = (lw < ls) and (rw < rs)
        left = (lw < ls) and not both
        right = (rw < rs) and not both
        if both: return "both_hands_up"
        if left: return "left_hand_up"
        if right: return "right_hand_up"
        return "idle"

    def run(self):
        pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        cap = cv2.VideoCapture(0)
        time.sleep(0.4)

        try:
            while self._running and cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                gesture = None
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    gesture = self.detect_gesture(results.pose_landmarks.landmark)
                    self.last_gestures.append(gesture)
                    # majority vote smoothing
                    gesture = collections.Counter(self.last_gestures).most_common(1)[0][0]
                else:
                    self.last_gestures.append("idle")
                    gesture = collections.Counter(self.last_gestures).most_common(1)[0][0]

                action_name = self.mapping.get(gesture or "idle", "none")
                target_key = key_from_action_name(action_name)
                self.update_keys(target_key)

                # overlay text
                cv2.putText(frame, f"Gesture: {gesture or 'none'}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(frame, f"Action:  {action_name}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                # convert to QImage (RGB)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                bytes_per_line = ch * w
                qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()
                self.frameReady.emit(qimg)
                self.status.emit(gesture or "none", action_name)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.update_keys(None)
            try:
                cap.release()
            except Exception:
                pass
            cv2.destroyAllWindows()

    def stop(self):
        self._running = False

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("pose.ui", self)
        # widgets
        self.cb_both  = self.findChild(QtWidgets.QComboBox, "comboBox_Both_hands_up")
        self.cb_left  = self.findChild(QtWidgets.QComboBox, "comboBox_Left_hand_up")
        self.cb_right = self.findChild(QtWidgets.QComboBox, "comboBox_Right_hand_up")
        self.cb_idle  = self.findChild(QtWidgets.QComboBox, "comboBox_Idle")
        self.btn_start = self.findChild(QtWidgets.QPushButton, "StartButton")
        self.btn_close = self.findChild(QtWidgets.QPushButton, "CloseButton")
        self.video_label = self.findChild(QtWidgets.QLabel, "videoLabel")
        self.gesture_label = self.findChild(QtWidgets.QLabel, "gestureLabel")
        self.action_label = self.findChild(QtWidgets.QLabel, "actionLabel")

        # populate combos
        for cb in (self.cb_both, self.cb_left, self.cb_right, self.cb_idle):
            cb.addItems(DEFAULT_ACTIONS)

        self.mapping = load_mapping()
        self.cb_both.setCurrentText(self.mapping.get("both_hands_up", "none"))
        self.cb_left.setCurrentText(self.mapping.get("left_hand_up", "none"))
        self.cb_right.setCurrentText(self.mapping.get("right_hand_up", "none"))
        self.cb_idle.setCurrentText(self.mapping.get("idle", "none"))

        self.btn_start.clicked.connect(self.on_start_stop)
        self.btn_close.clicked.connect(self.close)

        self.worker = None
        self.running = False
        self.last_frame = None

    def read_mapping_from_ui(self):
        return {
            "both_hands_up": self.cb_both.currentText(),
            "left_hand_up":  self.cb_left.currentText(),
            "right_hand_up": self.cb_right.currentText(),
            "idle":          self.cb_idle.currentText(),
        }

    def on_start_stop(self):
        if not self.running:
            self.mapping = self.read_mapping_from_ui()
            save_mapping(self.mapping)
            self.worker = PoseWorker(self.mapping)
            self.worker.frameReady.connect(self.on_frame)
            self.worker.status.connect(self.on_status)
            self.worker.start()
            self.running = True
            self.btn_start.setText("Stop")
            for cb in (self.cb_both, self.cb_left, self.cb_right, self.cb_idle):
                cb.setEnabled(False)
        else:
            if self.worker:
                self.worker.stop()
                self.worker.wait(1000)
            self.running = False
            self.btn_start.setText("Start")
            for cb in (self.cb_both, self.cb_left, self.cb_right, self.cb_idle):
                cb.setEnabled(True)

    @QtCore.pyqtSlot(QtGui.QImage)
    def on_frame(self, qimg):
        self.last_frame = qimg
        pix = QtGui.QPixmap.fromImage(qimg)
        # fit into label while keeping aspect ratio
        pix = pix.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)

    def resizeEvent(self, e):
        # re-fit last frame on resize
        if self.last_frame is not None:
            self.on_frame(self.last_frame)
        return super().resizeEvent(e)

    @QtCore.pyqtSlot(str, str)
    def on_status(self, gesture, action):
        self.gesture_label.setText(f"Gesture: {gesture}")
        self.action_label.setText(f"Action: {action}")

    def closeEvent(self, e):
        if self.worker and self.running:
            self.worker.stop()
            self.worker.wait(1000)
        return super().closeEvent(e)

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
