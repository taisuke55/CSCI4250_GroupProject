import sys, os, json, time, collections
import cv2
import mediapipe as mp
import pyautogui

from PyQt5 import QtWidgets, uic, QtCore, QtGui

CONFIG_PATH = "pose_command_map.json"

GESTURES = ["both_hands_up", "left_hand_up", "right_hand_up", "idle"]


def load_mapping():
    return {
        "both_hands_up": "up",
        "left_hand_up": "left",
        "right_hand_up": "right",
        "idle": "none",
    }


def save_mapping(mapping):
    return


def parse_action_string(action_str):
    base_key = "none"
    mods = {"shift": False, "ctrl": False, "alt": False}
    if not action_str:
        return base_key, mods
    s = action_str.strip().lower()
    if s == "" or s == "none":
        return "none", mods
    parts = [p.strip() for p in s.split("+") if p.strip()]
    for p in parts:
        if p in mods:
            mods[p] = True
        else:
            base_key = p
    if not base_key:
        base_key = "none"
    return base_key, mods


def build_action_string(base_key, mods):
    k = (base_key or "").strip().lower()
    if k == "" or k == "none":
        if not any(mods.values()):
            return "none"
        k = ""
    keys = []
    for m in ("ctrl", "alt", "shift"):
        if mods.get(m, False):
            keys.append(m)
    if k and k != "none":
        keys.append(k)
    if not keys:
        return "none"
    return "+".join(keys)


def keys_from_action_string(action_str):
    if not action_str:
        return set()
    s = action_str.strip().lower()
    if s == "" or s == "none":
        return set()
    parts = [p.strip() for p in s.split("+") if p.strip()]
    return set(parts)


class PoseWorker(QtCore.QThread):
    frameReady = QtCore.pyqtSignal(QtGui.QImage)
    status = QtCore.pyqtSignal(str, str)

    def __init__(self, mapping, parent=None):
        super().__init__(parent)
        self.mapping = mapping
        self._running = True
        self.current_keys = set()
        self.last_gestures = collections.deque(maxlen=10)
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

    def set_mapping(self, mapping: dict):
        self.mapping = mapping

    def update_keys(self, desired_keys):
        desired = set(desired_keys) if desired_keys else set()
        for k in desired - self.current_keys:
            pyautogui.keyDown(k)
        for k in self.current_keys - desired:
            pyautogui.keyUp(k)
        self.current_keys = desired

    def detect_gesture(self, landmarks):
        lw = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].y
        rw = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y
        ls = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        rs = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y

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
        pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        cap = cv2.VideoCapture(0)
        time.sleep(0.4)
        try:
            while self._running and cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                rgb_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_input)

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
                    self.last_gestures.append("idle")

                if self.last_gestures:
                    gesture = collections.Counter(
                        self.last_gestures
                    ).most_common(1)[0][0]
                else:
                    gesture = "idle"

                action_name = self.mapping.get(gesture or "idle", "none")
                desired_keys = keys_from_action_string(action_name)
                self.update_keys(desired_keys)

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
                self.frameReady.emit(qimg)
                self.status.emit(gesture or "none", action_name)
        finally:
            self.update_keys(None)
            try:
                cap.release()
            except Exception:
                pass
            cv2.destroyAllWindows()

    def stop(self):
        self._running = False


class SettingsWindow(QtWidgets.QMainWindow):
    mappingChosen = QtCore.pyqtSignal(dict)
    closed = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        uic.loadUi("pose_setting.ui", self)

        self.key_both = self.findChild(
            QtWidgets.QLineEdit, "keyEdit_Both_hands_up"
        )
        self.key_left = self.findChild(
            QtWidgets.QLineEdit, "keyEdit_Left_hand_up"
        )
        self.key_right = self.findChild(
            QtWidgets.QLineEdit, "keyEdit_Right_hand_up"
        )
        self.key_idle = self.findChild(
            QtWidgets.QLineEdit, "keyEdit_Idle"
        )

        self.chk_both_shift = self.findChild(
            QtWidgets.QCheckBox, "checkShift_Both"
        )
        self.chk_both_ctrl = self.findChild(
            QtWidgets.QCheckBox, "checkCtrl_Both"
        )
        self.chk_both_alt = self.findChild(
            QtWidgets.QCheckBox, "checkAlt_Both"
        )

        self.chk_left_shift = self.findChild(
            QtWidgets.QCheckBox, "checkShift_Left"
        )
        self.chk_left_ctrl = self.findChild(
            QtWidgets.QCheckBox, "checkCtrl_Left"
        )
        self.chk_left_alt = self.findChild(
            QtWidgets.QCheckBox, "checkAlt_Left"
        )

        self.chk_right_shift = self.findChild(
            QtWidgets.QCheckBox, "checkShift_Right"
        )
        self.chk_right_ctrl = self.findChild(
            QtWidgets.QCheckBox, "checkCtrl_Right"
        )
        self.chk_right_alt = self.findChild(
            QtWidgets.QCheckBox, "checkAlt_Right"
        )

        self.chk_idle_shift = self.findChild(
            QtWidgets.QCheckBox, "checkShift_Idle"
        )
        self.chk_idle_ctrl = self.findChild(
            QtWidgets.QCheckBox, "checkCtrl_Idle"
        )
        self.chk_idle_alt = self.findChild(
            QtWidgets.QCheckBox, "checkAlt_Idle"
        )

        self.btn_start = self.findChild(QtWidgets.QPushButton, "StartButton")
        self.btn_close = self.findChild(QtWidgets.QPushButton, "CloseButton")

        self.label_both_img = self.findChild(
            QtWidgets.QLabel, "label_both_img"
        )
        self.label_left_img = self.findChild(
            QtWidgets.QLabel, "label_left_img"
        )
        self.label_right_img = self.findChild(
            QtWidgets.QLabel, "label_right_img"
        )
        self.label_idle_img = self.findChild(
            QtWidgets.QLabel, "label_idle_img"
        )

        self.load_pose_images()

        for le in (self.key_both, self.key_left, self.key_right, self.key_idle):
            le.setReadOnly(True)
            le.installEventFilter(self)

        self._capturing_edit = None

        mapping = load_mapping()
        self.set_mapping(mapping)

        self.btn_start.clicked.connect(self.on_start_clicked)
        self.btn_close.clicked.connect(self.close)

    def load_pose_images(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        img_dir = os.path.join(base_dir, "img")

        def set_icon(label, filename):
            if label is None:
                return
            path = os.path.join(img_dir, filename)
            if not os.path.exists(path):
                return
            pix = QtGui.QPixmap(path)
            if pix.isNull():
                return
            size = label.size()
            if size.width() == 0 or size.height() == 0:
                w = h = 150
            else:
                w, h = size.width(), size.height()
            pix = pix.scaled(
                w, h,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            label.setPixmap(pix)

        set_icon(self.label_both_img, "both_hand.png")
        set_icon(self.label_left_img, "left_hand.png")
        set_icon(self.label_right_img, "right_hand.png")
        set_icon(self.label_idle_img, "idle.png")

    def set_mapping(self, mapping: dict):
        def apply(one_action, edit, chk_shift, chk_ctrl, chk_alt, default_key):
            base, mods = parse_action_string(one_action)
            if base == "none":
                base = default_key
            edit.setText(base)
            chk_shift.setChecked(mods["shift"])
            chk_ctrl.setChecked(mods["ctrl"])
            chk_alt.setChecked(mods["alt"])

        apply(mapping.get("both_hands_up", "up"),
              self.key_both,
              self.chk_both_shift, self.chk_both_ctrl, self.chk_both_alt,
              "up")

        apply(mapping.get("left_hand_up", "left"),
              self.key_left,
              self.chk_left_shift, self.chk_left_ctrl, self.chk_left_alt,
              "left")

        apply(mapping.get("right_hand_up", "right"),
              self.key_right,
              self.chk_right_shift, self.chk_right_ctrl, self.chk_right_alt,
              "right")

        apply(mapping.get("idle", "none"),
              self.key_idle,
              self.chk_idle_shift, self.chk_idle_ctrl, self.chk_idle_alt,
              "none")

    def read_mapping_from_ui(self):
        def get(edit, chk_shift, chk_ctrl, chk_alt, default_key):
            base = (edit.text() or "").strip().lower()
            if base == "":
                base = default_key
            mods = {
                "shift": chk_shift.isChecked(),
                "ctrl": chk_ctrl.isChecked(),
                "alt": chk_alt.isChecked(),
            }
            return build_action_string(base, mods)

        return {
            "both_hands_up": get(self.key_both,
                                 self.chk_both_shift,
                                 self.chk_both_ctrl,
                                 self.chk_both_alt,
                                 "up"),
            "left_hand_up": get(self.key_left,
                                self.chk_left_shift,
                                self.chk_left_ctrl,
                                self.chk_left_alt,
                                "left"),
            "right_hand_up": get(self.key_right,
                                 self.chk_right_shift,
                                 self.chk_right_ctrl,
                                 self.chk_right_alt,
                                 "right"),
            "idle": get(self.key_idle,
                        self.chk_idle_shift,
                        self.chk_idle_ctrl,
                        self.chk_idle_alt,
                        "none"),
        }

    def on_start_clicked(self):
        mapping = self.read_mapping_from_ui()
        save_mapping(mapping)
        self.mappingChosen.emit(mapping)

    def key_event_to_name(self, event: QtGui.QKeyEvent):
        key = event.key()
        if key == QtCore.Qt.Key_Escape:
            return "none"
        if QtCore.Qt.Key_A <= key <= QtCore.Qt.Key_Z:
            return chr(key).lower()
        if QtCore.Qt.Key_0 <= key <= QtCore.Qt.Key_9:
            return str(key - QtCore.Qt.Key_0)
        special = {
            QtCore.Qt.Key_Space: "space",
            QtCore.Qt.Key_Tab: "tab",
            QtCore.Qt.Key_Backspace: "backspace",
            QtCore.Qt.Key_Return: "enter",
            QtCore.Qt.Key_Enter: "enter",
            QtCore.Qt.Key_Up: "up",
            QtCore.Qt.Key_Down: "down",
            QtCore.Qt.Key_Left: "left",
            QtCore.Qt.Key_Right: "right",
            QtCore.Qt.Key_Shift: "shift",
            QtCore.Qt.Key_Control: "ctrl",
            QtCore.Qt.Key_Alt: "alt",
        }
        if key in special:
            return special[key]
        if QtCore.Qt.Key_F1 <= key <= QtCore.Qt.Key_F12:
            num = key - QtCore.Qt.Key_F1 + 1
            return f"f{num}"
        text = event.text()
        if text:
            return text.lower()
        return None

    def eventFilter(self, obj, event):
        edits = (self.key_both, self.key_left, self.key_right, self.key_idle)
        if obj in edits:
            if event.type() == QtCore.QEvent.MouseButtonPress:
                self._capturing_edit = obj
                obj.setText("Press a key...")
                return True
            if event.type() == QtCore.QEvent.KeyPress and self._capturing_edit is obj:
                name = self.key_event_to_name(event)
                if name is None:
                    name = "none"
                obj.setText(name)
                self._capturing_edit = None
                return True
        return super().eventFilter(obj, event)

    def closeEvent(self, e):
        self.closed.emit()
        return super().closeEvent(e)


class MainWindow(QtWidgets.QMainWindow):
    requestSettings = QtCore.pyqtSignal()

    def __init__(self, mapping: dict):
        super().__init__()
        uic.loadUi("pose.ui", self)

        initial_size = self.size()
        self.setMinimumSize(initial_size)

        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            rect = screen.availableGeometry()
            new_w = rect.width() // 2
            new_h = rect.height() // 2
            if new_w < initial_size.width():
                new_w = initial_size.width()
            if new_h < initial_size.height():
                new_h = initial_size.height()
            self.resize(new_w, new_h)
            self.move(
                rect.x() + (rect.width() - new_w) // 2,
                rect.y() + (rect.height() - new_h) // 2,
            )

        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)

        self.video_label = self.findChild(QtWidgets.QLabel, "videoLabel")

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

        self.aspect_offset = None
        self._resizing_from_code = False

        self.start_worker()

    def start_worker(self):
        if self.worker is not None:
            return
        self.worker = PoseWorker(self.mapping)
        self.worker.frameReady.connect(self.on_frame)
        self.worker.status.connect(self.on_status)
        self.worker.start()

    def stop_worker(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(1000)
            self.worker = None

    def update_mapping(self, mapping: dict):
        self.mapping = mapping
        if self.worker:
            self.worker.set_mapping(mapping)

    @QtCore.pyqtSlot()
    def on_setting_clicked(self):
        self.requestSettings.emit()

    @QtCore.pyqtSlot(QtGui.QImage)
    def on_frame(self, qimg):
        self.last_frame = qimg

        if self.aspect_offset is None:
            self.aspect_offset = self.height() - self.video_label.height()

        pix = QtGui.QPixmap.fromImage(qimg)
        pix = pix.scaled(
            self.video_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(pix)

    def resizeEvent(self, e):
        if self._resizing_from_code:
            self._resizing_from_code = False
        else:
            if self.last_frame is not None and self.aspect_offset is not None:
                frame_w = self.last_frame.width()
                frame_h = self.last_frame.height()
                if frame_w > 0 and frame_h > 0:
                    aspect = frame_h / frame_w
                    new_w = self.width()
                    new_video_h = int(new_w * aspect)
                    new_total_h = new_video_h + self.aspect_offset
                    if abs(new_total_h - self.height()) > 1:
                        self._resizing_from_code = True
                        self.resize(new_w, new_total_h)
                        return

        if self.last_frame is not None:
            self.on_frame(self.last_frame)

        return super().resizeEvent(e)

    @QtCore.pyqtSlot(str, str)
    def on_status(self, gesture, action):
        pass

    def closeEvent(self, e):
        self.stop_worker()
        return super().closeEvent(e)


class AppController(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self.settings = SettingsWindow()
        self.main = None

        self.settings.mappingChosen.connect(self.on_mapping_chosen)
        self.settings.closed.connect(self.on_settings_closed)

        self.settings.show()

    @QtCore.pyqtSlot(dict)
    def on_mapping_chosen(self, mapping: dict):
        if self.main is None:
            self.main = MainWindow(mapping)
            self.main.requestSettings.connect(self.on_request_settings)
        else:
            self.main.update_mapping(mapping)
        self.main.show()
        self.settings.hide()

    @QtCore.pyqtSlot()
    def on_request_settings(self):
        if self.main:
            self.settings.set_mapping(self.main.mapping)
        self.settings.show()

    @QtCore.pyqtSlot()
    def on_settings_closed(self):
        QtWidgets.QApplication.quit()


def main():
    app = QtWidgets.QApplication(sys.argv)
    controller = AppController()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
