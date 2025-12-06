import sys, os, json, time, collections, math
import cv2
import mediapipe as mp
import pyautogui

from PyQt5 import QtWidgets, uic, QtCore, QtGui

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "pose_command_map.json")

GESTURES = [
    "both_hands_up",
    "left_hand_up",
    "right_hand_up",
    "idle",
    "left_hand_up_high",
    "right_hand_up_high",
    "left_leg_up",
    "right_leg_up",
    "crouch",
]


def load_mapping():
    defaults = {
        "both_hands_up": "up",
        "left_hand_up": "left",
        "right_hand_up": "right",
        "idle": "none",
        "left_hand_up_high": "none",
        "right_hand_up_high": "none",
        "left_leg_up": "none",
        "right_leg_up": "none",
        "crouch": "none",
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            defaults.update(data)
        except Exception:
            pass
    return defaults


def save_mapping(mapping):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)
    except Exception:
        pass


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


def knee_angle(hip, knee, ankle):
    v1x = hip.x - knee.x
    v1y = hip.y - knee.y
    v2x = ankle.x - knee.x
    v2y = ankle.y - knee.y
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 == 0 or n2 == 0:
        return 180.0
    dot = v1x * v2x + v1y * v2y
    cos_theta = dot / (n1 * n2)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


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

        le = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y
        re = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].y

        lh = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        rh = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        lk = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
        rk = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        la = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        ra = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]

        left_wrist_up = lw < ls
        right_wrist_up = rw < rs
        both_hands = left_wrist_up and right_wrist_up

        left_elbow_low = le >= ls
        right_elbow_low = re >= rs
        left_elbow_high = le < ls
        right_elbow_high = re < rs

        left_knee_deg = knee_angle(lh, lk, la)
        right_knee_deg = knee_angle(rh, rk, ra)
        left_knee_bent = left_knee_deg < 160.0
        right_knee_bent = right_knee_deg < 160.0

        left_leg_up = la.y < ra.y - 0.05
        right_leg_up = ra.y < la.y - 0.05

        ankles_close = abs(la.y - ra.y) < 0.05
        crouch = left_knee_bent and right_knee_bent and ankles_close

        if crouch:
            return "crouch"

        if both_hands:
            return "both_hands_up"

        if left_leg_up and not right_leg_up:
            return "left_leg_up"
        if right_leg_up and not left_leg_up:
            return "right_leg_up"

        if left_wrist_up and not right_wrist_up:
            if left_elbow_low:
                return "left_hand_up"
            elif left_elbow_high:
                return "left_hand_up_high"

        if right_wrist_up and not left_wrist_up:
            if right_elbow_low:
                return "right_hand_up"
            elif right_elbow_high:
                return "right_hand_up_high"

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
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        rgb,
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

                # ---- Stylish HUD (left-top) with working font ----
                overlay = rgb.copy()
                cv2.rectangle(
                    overlay,
                    (10, 10),
                    (340, 80),
                    (0, 0, 0),
                    -1
                )
                cv2.addWeighted(overlay, 0.45, rgb, 0.55, 0, rgb)

                font = cv2.FONT_HERSHEY_DUPLEX

                cv2.putText(
                    rgb,
                    f"Gesture: {gesture}",
                    (20, 45),
                    font,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.putText(
                    rgb,
                    f"Action: {action_name}",
                    (20, 75),
                    font,
                    0.85,
                    (200, 200, 200),
                    2,
                    cv2.LINE_AA
                )
                # ----------------------------------------

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
                self.status.emit(gesture, action_name)
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
        uic.loadUi(os.path.join(BASE_DIR, "pose_setting.ui"), self)

        self.key_both = self.findChild(QtWidgets.QLineEdit, "keyEdit_Both_hands_up")
        self.key_left = self.findChild(QtWidgets.QLineEdit, "keyEdit_Left_hand_up")
        self.key_right = self.findChild(QtWidgets.QLineEdit, "keyEdit_Right_hand_up")
        self.key_idle = self.findChild(QtWidgets.QLineEdit, "keyEdit_Idle")
        self.key_left_high = self.findChild(QtWidgets.QLineEdit, "keyEdit_Left_hand_up_high")
        self.key_right_high = self.findChild(QtWidgets.QLineEdit, "keyEdit_Right_hand_up_high")
        self.key_left_leg = self.findChild(QtWidgets.QLineEdit, "keyEdit_Left_leg_up")
        self.key_right_leg = self.findChild(QtWidgets.QLineEdit, "keyEdit_Right_leg_up")
        self.key_crouch = self.findChild(QtWidgets.QLineEdit, "keyEdit_Crouch")

        self.label_both_img = self.findChild(QtWidgets.QLabel, "label_both_img")
        self.label_left_img = self.findChild(QtWidgets.QLabel, "label_left_img")
        self.label_right_img = self.findChild(QtWidgets.QLabel, "label_right_img")
        self.label_idle_img = self.findChild(QtWidgets.QLabel, "label_idle_img")
        self.label_left_high_img = self.findChild(QtWidgets.QLabel, "label_left_high_img")
        self.label_right_high_img = self.findChild(QtWidgets.QLabel, "label_right_high_img")
        self.label_left_leg_img = self.findChild(QtWidgets.QLabel, "label_left_leg_img")
        self.label_right_leg_img = self.findChild(QtWidgets.QLabel, "label_right_leg_img")
        self.label_crouch_img = self.findChild(QtWidgets.QLabel, "label_crouch_img")

        self.load_pose_images()

        mapping = load_mapping()
        self.set_mapping(mapping)

        self.btn_start = self.findChild(QtWidgets.QPushButton, "StartButton")
        self.btn_close = self.findChild(QtWidgets.QPushButton, "CloseButton")
        self.chk_save = self.findChild(QtWidgets.QCheckBox, "checkSaveForNext")

        edits = [
            self.key_both,
            self.key_left,
            self.key_right,
            self.key_idle,
            self.key_left_high,
            self.key_right_high,
            self.key_left_leg,
            self.key_right_leg,
            self.key_crouch,
        ]
        for le in edits:
            if le:
                le.setReadOnly(True)
                le.installEventFilter(self)

        self._capturing_edit = None

        self.btn_start.clicked.connect(self.on_start_clicked)
        self.btn_close.clicked.connect(self.close)

    def load_pose_images(self):
        img_dir = os.path.join(BASE_DIR, "img")

        def set_icon(label, filename):
            if not label:
                return
            path = os.path.join(img_dir, filename)
            if not os.path.exists(path):
                return
            pix = QtGui.QPixmap(path)
            if pix.isNull():
                return

            w = label.width()
            h = label.height()
            if w == 0 or h == 0:
                w = h = 150

            pix = pix.scaled(
                w, h,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            label.setPixmap(pix)

        set_icon(self.label_both_img, "both_hand.png")
        set_icon(self.label_left_img, "left_hand.png")
        set_icon(self.label_right_img, "right_hand.png")
        set_icon(self.label_idle_img, "idle.png")
        set_icon(self.label_left_high_img, "left_hand_high.png")
        set_icon(self.label_right_high_img, "right_hand_high.png")
        set_icon(self.label_left_leg_img, "left_leg_up.png")
        set_icon(self.label_right_leg_img, "right_leg_up.png")
        set_icon(self.label_crouch_img, "crouch.png")

    def set_mapping(self, mapping: dict):
        def apply(one_action, edit, shift, ctrl, alt, default_key):
            base, mods = parse_action_string(one_action)
            if base == "none":
                base = default_key
            edit.setText(base)
            shift.setChecked(mods["shift"])
            ctrl.setChecked(mods["ctrl"])
            alt.setChecked(mods["alt"])

        apply(mapping.get("both_hands_up", "up"),
              self.key_both, self.findChild(QtWidgets.QCheckBox,"checkShift_Both"),
              self.findChild(QtWidgets.QCheckBox,"checkCtrl_Both"),
              self.findChild(QtWidgets.QCheckBox,"checkAlt_Both"), "up")

        apply(mapping.get("left_hand_up", "left"),
              self.key_left, self.findChild(QtWidgets.QCheckBox,"checkShift_Left"),
              self.findChild(QtWidgets.QCheckBox,"checkCtrl_Left"),
              self.findChild(QtWidgets.QCheckBox,"checkAlt_Left"), "left")

        apply(mapping.get("right_hand_up", "right"),
              self.key_right, self.findChild(QtWidgets.QCheckBox,"checkShift_Right"),
              self.findChild(QtWidgets.QCheckBox,"checkCtrl_Right"),
              self.findChild(QtWidgets.QCheckBox,"checkAlt_Right"), "right")

        apply(mapping.get("idle", "none"),
              self.key_idle, self.findChild(QtWidgets.QCheckBox,"checkShift_Idle"),
              self.findChild(QtWidgets.QCheckBox,"checkCtrl_Idle"),
              self.findChild(QtWidgets.QCheckBox,"checkAlt_Idle"), "none")

        apply(mapping.get("left_hand_up_high", "none"),
              self.key_left_high, self.findChild(QtWidgets.QCheckBox,"checkShift_LeftHigh"),
              self.findChild(QtWidgets.QCheckBox,"checkCtrl_LeftHigh"),
              self.findChild(QtWidgets.QCheckBox,"checkAlt_LeftHigh"), "none")

        apply(mapping.get("right_hand_up_high", "none"),
              self.key_right_high, self.findChild(QtWidgets.QCheckBox,"checkShift_RightHigh"),
              self.findChild(QtWidgets.QCheckBox,"checkCtrl_RightHigh"),
              self.findChild(QtWidgets.QCheckBox,"checkAlt_RightHigh"), "none")

        apply(mapping.get("left_leg_up", "none"),
              self.key_left_leg, self.findChild(QtWidgets.QCheckBox,"checkShift_LeftLeg"),
              self.findChild(QtWidgets.QCheckBox,"checkCtrl_LeftLeg"),
              self.findChild(QtWidgets.QCheckBox,"checkAlt_LeftLeg"), "none")

        apply(mapping.get("right_leg_up", "none"),
              self.key_right_leg, self.findChild(QtWidgets.QCheckBox,"checkShift_RightLeg"),
              self.findChild(QtWidgets.QCheckBox,"checkCtrl_RightLeg"),
              self.findChild(QtWidgets.QCheckBox,"checkAlt_RightLeg"), "none")

        apply(mapping.get("crouch", "none"),
              self.key_crouch, self.findChild(QtWidgets.QCheckBox,"checkShift_Crouch"),
              self.findChild(QtWidgets.QCheckBox,"checkCtrl_Crouch"),
              self.findChild(QtWidgets.QCheckBox,"checkAlt_Crouch"), "none")

    def read_mapping_from_ui(self):
        def get(edit, shift, ctrl, alt, default_key):
            base = (edit.text() or "").strip().lower()
            if base == "":
                base = default_key
            mods = {
                "shift": shift.isChecked(),
                "ctrl": ctrl.isChecked(),
                "alt": alt.isChecked(),
            }
            return build_action_string(base, mods)

        return {
            "both_hands_up": get(self.key_both,
                                 self.findChild(QtWidgets.QCheckBox,"checkShift_Both"),
                                 self.findChild(QtWidgets.QCheckBox,"checkCtrl_Both"),
                                 self.findChild(QtWidgets.QCheckBox,"checkAlt_Both"),
                                 "up"),
            "left_hand_up": get(self.key_left,
                                self.findChild(QtWidgets.QCheckBox,"checkShift_Left"),
                                self.findChild(QtWidgets.QCheckBox,"checkCtrl_Left"),
                                self.findChild(QtWidgets.QCheckBox,"checkAlt_Left"),
                                "left"),
            "right_hand_up": get(self.key_right,
                                 self.findChild(QtWidgets.QCheckBox,"checkShift_Right"),
                                 self.findChild(QtWidgets.QCheckBox,"checkCtrl_Right"),
                                 self.findChild(QtWidgets.QCheckBox,"checkAlt_Right"),
                                 "right"),
            "idle": get(self.key_idle,
                        self.findChild(QtWidgets.QCheckBox,"checkShift_Idle"),
                        self.findChild(QtWidgets.QCheckBox,"checkCtrl_Idle"),
                        self.findChild(QtWidgets.QCheckBox,"checkAlt_Idle"),
                        "none"),
            "left_hand_up_high": get(self.key_left_high,
                                     self.findChild(QtWidgets.QCheckBox,"checkShift_LeftHigh"),
                                     self.findChild(QtWidgets.QCheckBox,"checkCtrl_LeftHigh"),
                                     self.findChild(QtWidgets.QCheckBox,"checkAlt_LeftHigh"),
                                     "none"),
            "right_hand_up_high": get(self.key_right_high,
                                      self.findChild(QtWidgets.QCheckBox,"checkShift_RightHigh"),
                                      self.findChild(QtWidgets.QCheckBox,"checkCtrl_RightHigh"),
                                      self.findChild(QtWidgets.QCheckBox,"checkAlt_RightHigh"),
                                      "none"),
            "left_leg_up": get(self.key_left_leg,
                               self.findChild(QtWidgets.QCheckBox,"checkShift_LeftLeg"),
                               self.findChild(QtWidgets.QCheckBox,"checkCtrl_LeftLeg"),
                               self.findChild(QtWidgets.QCheckBox,"checkAlt_LeftLeg"),
                               "none"),
            "right_leg_up": get(self.key_right_leg,
                                self.findChild(QtWidgets.QCheckBox,"checkShift_RightLeg"),
                                self.findChild(QtWidgets.QCheckBox,"checkCtrl_RightLeg"),
                                self.findChild(QtWidgets.QCheckBox,"checkAlt_RightLeg"),
                                "none"),
            "crouch": get(self.key_crouch,
                          self.findChild(QtWidgets.QCheckBox,"checkShift_Crouch"),
                          self.findChild(QtWidgets.QCheckBox,"checkCtrl_Crouch"),
                          self.findChild(QtWidgets.QCheckBox,"checkAlt_Crouch"),
                          "none"),
        }

    def on_start_clicked(self):
        mapping = self.read_mapping_from_ui()
        if self.chk_save and self.chk_save.isChecked():
            save_mapping(mapping)
        self.mappingChosen.emit(mapping)

    def eventFilter(self, obj, event):
        edits = [
            self.key_both,
            self.key_left,
            self.key_right,
            self.key_idle,
            self.key_left_high,
            self.key_right_high,
            self.key_left_leg,
            self.key_right_leg,
            self.key_crouch,
        ]
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

    def key_event_to_name(self, event):
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

    def closeEvent(self, e):
        self.closed.emit()
        return super().closeEvent(e)


class MainWindow(QtWidgets.QMainWindow):
    requestSettings = QtCore.pyqtSignal()

    def __init__(self, mapping: dict):
        super().__init__()
        uic.loadUi(os.path.join(BASE_DIR, "pose.ui"), self)

        self.video_label = self.findChild(QtWidgets.QLabel, "videoLabel")

        self.btn_setting = self.findChild(QtWidgets.QPushButton, "Setting_Button")
        self.btn_close = self.findChild(QtWidgets.QPushButton, "Close_Button")

        self.btn_setting.clicked.connect(self.on_setting_clicked)
        self.btn_close.clicked.connect(self.close)

        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)

        self.frame_aspect = None

        self.mapping = mapping
        self.worker = PoseWorker(mapping)
        self.worker.frameReady.connect(self.on_frame)
        self.worker.status.connect(self.on_status)
        self.worker.start()

        self.show()
        self.resize(960, 720)
        self.video_label.setMinimumSize(640, 480)

    def on_setting_clicked(self):
        self.requestSettings.emit()

    @QtCore.pyqtSlot(QtGui.QImage)
    def on_frame(self, qimg):
        if self.frame_aspect is None:
            w = qimg.width()
            h = qimg.height()
            if h != 0:
                self.frame_aspect = w / h

        pix = QtGui.QPixmap.fromImage(qimg)
        pix = pix.scaled(
            self.video_label.size(),
            QtCore.Qt.KeepAspectRatioByExpanding,
            QtCore.Qt.SmoothTransformation,
        )

        self.video_label.setPixmap(pix)

    @QtCore.pyqtSlot(str, str)
    def on_status(self, g, a):
        pass

    def closeEvent(self, e):
        if self.worker:
            self.worker.stop()
            self.worker.wait(500)
        return super().closeEvent(e)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)

        if not self.frame_aspect:
            return

        rect = self.centralWidget().contentsRect()
        avail_w = rect.width()
        avail_h = rect.height()

        buttons_height = self.btn_setting.height()
        avail_h = max(0, avail_h - buttons_height - 20)

        if avail_w <= 0 or avail_h <= 0:
            return

        target_w = avail_w
        target_h = int(target_w / self.frame_aspect)

        if target_h > avail_h:
            target_h = avail_h
            target_w = int(target_h * self.frame_aspect)

        self.video_label.setMinimumSize(target_w, target_h)
        self.video_label.setMaximumSize(target_w, target_h)
        self.video_label.resize(target_w, target_h)



class AppController(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self.settings = SettingsWindow()
        self.settings.mappingChosen.connect(self.on_mapping_chosen)
        self.settings.closed.connect(self.on_app_quit)
        self.settings.show()
        self.main = None

    def on_mapping_chosen(self, mapping):
        if self.main is None:
            self.main = MainWindow(mapping)
            self.main.requestSettings.connect(self.on_request_settings)
        else:
            self.main.mapping = mapping
            self.main.worker.set_mapping(mapping)
        self.main.show()
        self.settings.hide()

    def on_request_settings(self):
        if self.main:
            m = self.main.mapping
            self.settings.set_mapping(m)
        self.settings.show()

    def on_app_quit(self):
        QtWidgets.QApplication.quit()


def main():
    app = QtWidgets.QApplication(sys.argv)
    controller = AppController()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
