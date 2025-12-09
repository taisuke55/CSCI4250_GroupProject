import sys, os, json, time, collections, math
import cv2
import mediapipe as mp
import pyautogui

from PyQt5 import QtWidgets, uic, QtCore, QtGui

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "pose_command_map.json")

# Supported gesture identifiers
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

# Pretty names (gesture display in HUD)
PRETTY_GESTURE_NAME = {
    "both_hands_up": "Both hands up",
    "left_hand_up": "Left hand up",
    "right_hand_up": "Right hand up",
    "idle": "Idle",
    "left_hand_up_high": "Left hand up (high elbow)",
    "right_hand_up_high": "Right hand up (high elbow)",
    "left_leg_up": "Left leg up",
    "right_leg_up": "Right leg up",
    "crouch": "Crouch",
}

# Pretty names (action keys display in HUD)
PRETTY_KEY_NAME = {
    "left": "Left arrow",
    "right": "Right arrow",
    "up": "Up arrow",
    "down": "Down arrow",
    "space": "Space",
    "shift": "Shift",
    "ctrl": "Ctrl",
    "alt": "Alt",
}

# Threshold (in normalized y difference) to decide leg is "up".
# Smaller value → easier to detect raised leg.
LEG_LIFT_THRESHOLD = 0.03


# =============================================================
#                 CONFIG (LOAD / SAVE)
# =============================================================

def load_mapping():
    """
    Load saved gesture→action mapping from JSON.
    If mapping file does not exist, load defaults.
    """
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
            # If loading fails, just use defaults
            pass
    return defaults


def save_mapping(mapping):
    """Write mapping dictionary to JSON file."""
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)
    except Exception:
        # Fail silently – mapping is not critical data
        pass


# =============================================================
#                 STRING ↔ ACTION HELPERS
# =============================================================

def parse_action_string(action_str):
    """
    Parse "ctrl+space" → (base="space", mods={"ctrl":True,"shift":False,...})
    Used to populate UI fields and modifier checkboxes.
    """
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
    """
    Combine base key and modifier bools → "ctrl+left".
    Used when reading back from the UI fields into mapping.
    """
    k = (base_key or "").strip().lower()
    if k == "" or k == "none":
        # No base key: if no modifiers either, this means "none"
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
    """
    Convert "left+space" → {"left","space"} for simultaneous key presses.
    If action is "none", returns empty set (no key pressed).
    """
    if not action_str:
        return set()
    s = action_str.strip().lower()
    if s == "" or s == "none":
        return set()
    parts = [p.strip() for p in s.split("+") if p.strip()]
    return set(parts)


# =============================================================
#                      POSE HELPERS
# =============================================================

def knee_angle(hip, knee, ankle):
    """
    Calculate knee joint angle in degrees using vector math.
    Smaller angle → more bent knee.
    """
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


# =============================================================
#                     POSE WORKER THREAD
# =============================================================

class PoseWorker(QtCore.QThread):
    """
    Capture camera frames, run MediaPipe pose detection,
    determine gestures, and simulate keyboard input.
    """
    frameReady = QtCore.pyqtSignal(QtGui.QImage)
    status = QtCore.pyqtSignal(str, str)

    def __init__(self, mapping, parent=None):
        super().__init__(parent)
        self.mapping = mapping
        self._running = True
        self.current_keys = set()
        # For smoothing: store last N primary gestures
        self.last_gestures = collections.deque(maxlen=10)
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

    # -----------------------
    # Key simulation
    # -----------------------
    def set_mapping(self, mapping: dict):
        """Update gesture→action mapping while running."""
        self.mapping = mapping

    def update_keys(self, desired_keys):
        """
        Update which keys are currently being held down.

        desired_keys: set of key names such as {"left", "space"}.
        """
        desired = set(desired_keys) if desired_keys else set()

        # Press new keys
        for k in desired - self.current_keys:
            pyautogui.keyDown(k)

        # Release keys no longer needed
        for k in self.current_keys - desired:
            pyautogui.keyUp(k)

        self.current_keys = desired

    # -----------------------
    # Gesture detection
    # -----------------------
    def detect_gesture(self, landmarks):
        """
        Determine all simultaneous gestures detected in this frame.

        Returns:
            primary: str      (gesture used as "main" for smoothing/debug)
            flags: dict       (all gestures that are currently active,
                               e.g., {"left_hand_up":True,"left_leg_up":True,...})
        """
        # Hands: note we flip the frame horizontally, so RIGHT_* appears
        # on the left side of the screen visually (user's right hand).
        lw = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].y
        rw = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y
        ls = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        rs = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y

        le = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y
        re = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].y

        # Legs: also mirror to match what the user sees on screen.
        # Because the frame is flipped horizontally:
        #   RIGHT_* landmarks appear on the left side of the screen
        #   LEFT_*  landmarks appear on the right side of the screen
        lh = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]   # screen-left leg
        rh = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]    # screen-right leg
        lk = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        rk = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
        la = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        ra = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]

        left_knee_deg = knee_angle(lh, lk, la)
        right_knee_deg = knee_angle(rh, rk, ra)
        left_knee_bent = left_knee_deg < 160.0
        right_knee_bent = right_knee_deg < 160.0

        # Leg up detection: if one ankle is higher (smaller y) than the other
        # by LEG_LIFT_THRESHOLD or more.
        left_leg_up = la.y < ra.y - LEG_LIFT_THRESHOLD
        right_leg_up = ra.y < la.y - LEG_LIFT_THRESHOLD

        ankles_close = abs(la.y - ra.y) < 0.05
        crouch = left_knee_bent and right_knee_bent and ankles_close

        # Basic conditions for arms
        left_wrist_up = lw < ls
        right_wrist_up = rw < rs
        both_hands = left_wrist_up and right_wrist_up

        left_elbow_low = le >= ls
        right_elbow_low = re >= rs
        left_elbow_high = le < ls
        right_elbow_high = re < rs

        # Initialize flags for all gestures
        flags = {
            "both_hands_up": False,
            "left_hand_up": False,
            "right_hand_up": False,
            "left_hand_up_high": False,
            "right_hand_up_high": False,
            "left_leg_up": False,
            "right_leg_up": False,
            "crouch": False,
            "idle": False,
        }

        # Legs and crouch
        if crouch:
            flags["crouch"] = True

        if left_leg_up and not right_leg_up:
            flags["left_leg_up"] = True
        elif right_leg_up and not left_leg_up:
            flags["right_leg_up"] = True

        # Arms
        if both_hands:
            flags["both_hands_up"] = True
        else:
            if left_wrist_up and not right_wrist_up:
                if left_elbow_low:
                    flags["left_hand_up"] = True
                elif left_elbow_high:
                    flags["left_hand_up_high"] = True
            if right_wrist_up and not left_wrist_up:
                if right_elbow_low:
                    flags["right_hand_up"] = True
                elif right_elbow_high:
                    flags["right_hand_up_high"] = True

        # Determine primary gesture using priority order
        if flags["crouch"]:
            primary = "crouch"
        elif flags["both_hands_up"]:
            primary = "both_hands_up"
        elif flags["left_leg_up"]:
            primary = "left_leg_up"
        elif flags["right_leg_up"]:
            primary = "right_leg_up"
        elif flags["left_hand_up_high"]:
            primary = "left_hand_up_high"
        elif flags["right_hand_up_high"]:
            primary = "right_hand_up_high"
        elif flags["left_hand_up"]:
            primary = "left_hand_up"
        elif flags["right_hand_up"]:
            primary = "right_hand_up"
        else:
            # No gesture detected → idle
            primary = "idle"
            flags["idle"] = True

        return primary, flags

    # -----------------------
    # Visibility check
    # -----------------------
    def is_full_body_visible(self, landmarks):
        """
        Heuristic check: are major joints present and visible?
        Used to show a warning when pose detection confidence is low.
        """
        pose_lm = self.mp_pose.PoseLandmark

        required = [
            pose_lm.LEFT_SHOULDER,
            pose_lm.RIGHT_SHOULDER,
            pose_lm.LEFT_HIP,
            pose_lm.RIGHT_HIP,
            pose_lm.LEFT_KNEE,
            pose_lm.RIGHT_KNEE,
            pose_lm.LEFT_ANKLE,
            pose_lm.RIGHT_ANKLE,
        ]

        for r in required:
            lm = landmarks[r]
            if lm.visibility < 0.5 or lm.y < 0 or lm.y > 1:
                return False
        return True

    # -----------------------
    # Thread loop
    # -----------------------
    def run(self):
        """
        Main worker loop:
        - Capture camera
        - Detect pose / gestures
        - Build combined actions from hands + legs
        - Update HUD
        - Emit video frames to main UI
        """
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

                # Mirror image
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                full_body_warning = ""
                active_flags = None

                if results.pose_landmarks:
                    # Draw skeleton
                    self.mp_drawing.draw_landmarks(
                        rgb,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                    )

                    landmarks = results.pose_landmarks.landmark
                    primary, flags = self.detect_gesture(landmarks)

                    # Keep track of primary gesture for smoothing / debug
                    self.last_gestures.append(primary)
                    active_flags = flags

                    if not self.is_full_body_visible(landmarks):
                        full_body_warning = "Please show your whole body to the camera"
                else:
                    # No landmarks → treat as idle
                    self.last_gestures.append("idle")
                    active_flags = {"idle": True}
                    full_body_warning = "Please show your whole body to the camera"

                # Smoothed primary gesture (used for fallback / debug)
                if self.last_gestures:
                    gesture = collections.Counter(
                        self.last_gestures
                    ).most_common(1)[0][0]
                else:
                    gesture = "idle"

                # ------------------------------------------------
                # Build combined actions from hand and leg groups
                # ------------------------------------------------
                desired_keys = set()

                if active_flags is None:
                    active_flags = {}

                # Priority order for hand gestures
                hand_priority = [
                    "both_hands_up",
                    "left_hand_up_high",
                    "left_hand_up",
                    "right_hand_up_high",
                    "right_hand_up",
                ]

                # Priority order for leg / body gestures
                leg_priority = [
                    "crouch",
                    "left_leg_up",
                    "right_leg_up",
                ]

                # Pick one active hand gesture (if any)
                active_hand = None
                for g in hand_priority:
                    if active_flags.get(g):
                        active_hand = g
                        break

                # Pick one active leg gesture (if any)
                active_leg = None
                for g in leg_priority:
                    if active_flags.get(g):
                        active_leg = g
                        break

                # Combine actions from hand + leg
                if active_hand:
                    s = self.mapping.get(active_hand, "none")
                    if s != "none":
                        desired_keys |= keys_from_action_string(s)

                if active_leg:
                    s = self.mapping.get(active_leg, "none")
                    if s != "none":
                        desired_keys |= keys_from_action_string(s)

                # If nothing from hand/leg produced an action, fall back to smoothed primary
                if not desired_keys:
                    fallback_action = self.mapping.get(gesture or "idle", "none")
                    if fallback_action != "none":
                        desired_keys = keys_from_action_string(fallback_action)

                # Finally update key presses
                self.update_keys(desired_keys)

                # -------------------------------
                # HUD overlay (gesture + action)
                # -------------------------------
                overlay = rgb.copy()
                cv2.rectangle(
                    overlay,
                    (10, 10),
                    (520, 130),
                    (0, 0, 0),
                    -1
                )
                cv2.addWeighted(overlay, 0.45, rgb, 0.55, 0, rgb)

                font = cv2.FONT_HERSHEY_DUPLEX

                # ---- Display all simultaneous gestures in this frame ----
                pretty_gesture = None
                display_names = []

                if active_flags:
                    # List all active gestures except "idle"
                    for g, on in active_flags.items():
                        if on and g != "idle" and g in PRETTY_GESTURE_NAME:
                            display_names.append(PRETTY_GESTURE_NAME[g])

                if display_names:
                    # e.g., "Right hand up + Left leg up"
                    pretty_gesture = " + ".join(display_names)
                else:
                    # Fallback to smoothed primary gesture
                    pretty_gesture = PRETTY_GESTURE_NAME.get(gesture, gesture)

                cv2.putText(
                    rgb,
                    f"Gesture: {pretty_gesture}",
                    (20, 45),
                    font,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                # ---- Display all keys in the combined action ----
                if desired_keys:
                    # Sort keys to show modifiers first, then others
                    order = ["ctrl", "alt", "shift", "left", "right", "up", "down", "space"]
                    sorted_keys = sorted(
                        list(desired_keys),
                        key=lambda k: order.index(k) if k in order else len(order)
                    )
                    display_keys = [PRETTY_KEY_NAME.get(p, p) for p in sorted_keys]
                    action_text = " + ".join(display_keys)
                else:
                    action_text = "none"

                cv2.putText(
                    rgb,
                    f"Action: {action_text}",
                    (20, 75),
                    font,
                    0.85,
                    (200, 200, 200),
                    2,
                    cv2.LINE_AA
                )

                if full_body_warning:
                    cv2.putText(
                        rgb,
                        full_body_warning,
                        (20, 105),
                        font,
                        0.75,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                # ---- Send frame to Qt UI ----
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
                # For status, we emit the smoothed primary gesture and joined keys
                joined_keys = "+".join(sorted(desired_keys)) if desired_keys else "none"
                self.status.emit(gesture, joined_keys)

        finally:
            # Release all held keys on stop
            self.update_keys(None)
            try:
                cap.release()
            except Exception:
                pass
            cv2.destroyAllWindows()

    def stop(self):
        """Stop thread loop."""
        self._running = False


# =============================================================
#                SETTINGS WINDOW (Qt UI)
# =============================================================

class SettingsWindow(QtWidgets.QMainWindow):
    """
    Settings window:
    - Shows gesture images
    - Lets user assign keys to each gesture
    - Can optionally save mapping to JSON
    """
    mappingChosen = QtCore.pyqtSignal(dict)
    closed = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        uic.loadUi(os.path.join(BASE_DIR, "pose_setting.ui"), self)

        # Key input fields
        self.key_both = self.findChild(QtWidgets.QLineEdit, "keyEdit_Both_hands_up")
        self.key_left = self.findChild(QtWidgets.QLineEdit, "keyEdit_Left_hand_up")
        self.key_right = self.findChild(QtWidgets.QLineEdit, "keyEdit_Right_hand_up")
        self.key_idle = self.findChild(QtWidgets.QLineEdit, "keyEdit_Idle")
        self.key_left_high = self.findChild(QtWidgets.QLineEdit, "keyEdit_Left_hand_up_high")
        self.key_right_high = self.findChild(QtWidgets.QLineEdit, "keyEdit_Right_hand_up_high")
        self.key_left_leg = self.findChild(QtWidgets.QLineEdit, "keyEdit_Left_leg_up")
        self.key_right_leg = self.findChild(QtWidgets.QLineEdit, "keyEdit_Right_leg_up")
        self.key_crouch = self.findChild(QtWidgets.QLineEdit, "keyEdit_Crouch")

        # Pose image labels
        self.label_both_img = self.findChild(QtWidgets.QLabel, "label_both_img")
        self.label_left_img = self.findChild(QtWidgets.QLabel, "label_left_img")
        self.label_right_img = self.findChild(QtWidgets.QLabel, "label_right_img")
        self.label_idle_img = self.findChild(QtWidgets.QLabel, "label_idle_img")
        self.label_left_high_img = self.findChild(QtWidgets.QLabel, "label_left_high_img")
        self.label_right_high_img = self.findChild(QtWidgets.QLabel, "label_right_high_img")
        self.label_left_leg_img = self.findChild(QtWidgets.QLabel, "label_left_leg_img")
        self.label_right_leg_img = self.findChild(QtWidgets.QLabel, "label_right_leg_img")
        self.label_crouch_img = self.findChild(QtWidgets.QLabel, "label_crouch_img")

        # Load PNG silhouette images
        self.load_pose_images()

        # Load existing mapping and apply to UI
        mapping = load_mapping()
        self.set_mapping(mapping)

        # Buttons and checkbox
        self.btn_start = self.findChild(QtWidgets.QPushButton, "StartButton")
        self.btn_close = self.findChild(QtWidgets.QPushButton, "CloseButton")
        self.chk_save = self.findChild(QtWidgets.QCheckBox, "checkSaveForNext")

        # Make all key fields readonly and capture key events via eventFilter
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

        # Connect buttons
        self.btn_start.clicked.connect(self.on_start_clicked)
        self.btn_close.clicked.connect(self.close)

        # Center this window on screen after it is shown
        QtCore.QTimer.singleShot(0, self.center_on_screen)

    def center_on_screen(self):
        """Move this window to the center of the primary screen."""
        screen = QtWidgets.QApplication.primaryScreen()
        if not screen:
            return
        screen_geo = screen.availableGeometry()
        geo = self.frameGeometry()
        geo.moveCenter(screen_geo.center())
        self.move(geo.topLeft())

    def load_pose_images(self):
        """Load and scale pose images into the labels."""
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
                # Fallback default size when not yet laid out
                w = h = 150

            pix = pix.scaled(
                w, h,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            label.setPixmap(pix)

        # These filenames follow the user's preferred naming
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
        """
        Apply mapping dict to UI fields and modifier checkboxes.
        """
        def apply(one_action, edit, shift, ctrl, alt, default_key):
            base, mods = parse_action_string(one_action)
            # If mapping is "none", still show a default base key in the box
            if base == "none":
                base = default_key
            edit.setText(base)
            shift.setChecked(mods["shift"])
            ctrl.setChecked(mods["ctrl"])
            alt.setChecked(mods["alt"])

        apply(mapping.get("both_hands_up", "up"),
              self.key_both, self.findChild(QtWidgets.QCheckBox, "checkShift_Both"),
              self.findChild(QtWidgets.QCheckBox, "checkCtrl_Both"),
              self.findChild(QtWidgets.QCheckBox, "checkAlt_Both"), "up")

        apply(mapping.get("left_hand_up", "left"),
              self.key_left, self.findChild(QtWidgets.QCheckBox, "checkShift_Left"),
              self.findChild(QtWidgets.QCheckBox, "checkCtrl_Left"),
              self.findChild(QtWidgets.QCheckBox, "checkAlt_Left"), "left")

        apply(mapping.get("right_hand_up", "right"),
              self.key_right, self.findChild(QtWidgets.QCheckBox, "checkShift_Right"),
              self.findChild(QtWidgets.QCheckBox, "checkCtrl_Right"),
              self.findChild(QtWidgets.QCheckBox, "checkAlt_Right"), "right")

        apply(mapping.get("idle", "none"),
              self.key_idle, self.findChild(QtWidgets.QCheckBox, "checkShift_Idle"),
              self.findChild(QtWidgets.QCheckBox, "checkCtrl_Idle"),
              self.findChild(QtWidgets.QCheckBox, "checkAlt_Idle"), "none")

        apply(mapping.get("left_hand_up_high", "none"),
              self.key_left_high, self.findChild(QtWidgets.QCheckBox, "checkShift_LeftHigh"),
              self.findChild(QtWidgets.QCheckBox, "checkCtrl_LeftHigh"),
              self.findChild(QtWidgets.QCheckBox, "checkAlt_LeftHigh"), "none")

        apply(mapping.get("right_hand_up_high", "none"),
              self.key_right_high, self.findChild(QtWidgets.QCheckBox, "checkShift_RightHigh"),
              self.findChild(QtWidgets.QCheckBox, "checkCtrl_RightHigh"),
              self.findChild(QtWidgets.QCheckBox, "checkAlt_RightHigh"), "none")

        apply(mapping.get("left_leg_up", "none"),
            self.key_left_leg, self.findChild(QtWidgets.QCheckBox, "checkShift_LeftLeg"),
            self.findChild(QtWidgets.QCheckBox, "checkCtrl_LeftLeg"),
            self.findChild(QtWidgets.QCheckBox, "checkAlt_LeftLeg"), "none")

        apply(mapping.get("right_leg_up", "none"),
            self.key_right_leg, self.findChild(QtWidgets.QCheckBox, "checkShift_RightLeg"),
            self.findChild(QtWidgets.QCheckBox, "checkCtrl_RightLeg"),
            self.findChild(QtWidgets.QCheckBox, "checkAlt_RightLeg"), "none")

        apply(mapping.get("crouch", "none"),
              self.key_crouch, self.findChild(QtWidgets.QCheckBox, "checkShift_Crouch"),
              self.findChild(QtWidgets.QCheckBox, "checkCtrl_Crouch"),
              self.findChild(QtWidgets.QCheckBox, "checkAlt_Crouch"), "none")

    def read_mapping_from_ui(self):
        """
        Read values from UI fields and modifier checkboxes
        and build a gesture→action mapping dict.
        """
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
                                 self.findChild(QtWidgets.QCheckBox, "checkShift_Both"),
                                 self.findChild(QtWidgets.QCheckBox, "checkCtrl_Both"),
                                 self.findChild(QtWidgets.QCheckBox, "checkAlt_Both"),
                                 "up"),
            "left_hand_up": get(self.key_left,
                                self.findChild(QtWidgets.QCheckBox, "checkShift_Left"),
                                self.findChild(QtWidgets.QCheckBox, "checkCtrl_Left"),
                                self.findChild(QtWidgets.QCheckBox, "checkAlt_Left"),
                                "left"),
            "right_hand_up": get(self.key_right,
                                 self.findChild(QtWidgets.QCheckBox, "checkShift_Right"),
                                 self.findChild(QtWidgets.QCheckBox, "checkCtrl_Right"),
                                 self.findChild(QtWidgets.QCheckBox, "checkAlt_Right"),
                                 "right"),
            "idle": get(self.key_idle,
                        self.findChild(QtWidgets.QCheckBox, "checkShift_Idle"),
                        self.findChild(QtWidgets.QCheckBox, "checkCtrl_Idle"),
                        self.findChild(QtWidgets.QCheckBox, "checkAlt_Idle"),
                        "none"),
            "left_hand_up_high": get(self.key_left_high,
                                     self.findChild(QtWidgets.QCheckBox, "checkShift_LeftHigh"),
                                     self.findChild(QtWidgets.QCheckBox, "checkCtrl_LeftHigh"),
                                     self.findChild(QtWidgets.QCheckBox, "checkAlt_LeftHigh"),
                                     "none"),
            "right_hand_up_high": get(self.key_right_high,
                                      self.findChild(QtWidgets.QCheckBox, "checkShift_RightHigh"),
                                      self.findChild(QtWidgets.QCheckBox, "checkCtrl_RightHigh"),
                                      self.findChild(QtWidgets.QCheckBox, "checkAlt_RightHigh"),
                                      "none"),
            "left_leg_up": get(self.key_left_leg,
                               self.findChild(QtWidgets.QCheckBox, "checkShift_LeftLeg"),
                               self.findChild(QtWidgets.QCheckBox, "checkCtrl_LeftLeg"),
                               self.findChild(QtWidgets.QCheckBox, "checkAlt_LeftLeg"),
                               "none"),
            "right_leg_up": get(self.key_right_leg,
                                self.findChild(QtWidgets.QCheckBox, "checkShift_RightLeg"),
                                self.findChild(QtWidgets.QCheckBox, "checkCtrl_RightLeg"),
                                self.findChild(QtWidgets.QCheckBox, "checkAlt_RightLeg"),
                                "none"),
            "crouch": get(self.key_crouch,
                          self.findChild(QtWidgets.QCheckBox, "checkShift_Crouch"),
                          self.findChild(QtWidgets.QCheckBox, "checkCtrl_Crouch"),
                          self.findChild(QtWidgets.QCheckBox, "checkAlt_Crouch"),
                          "none"),
        }

    def on_start_clicked(self):
        """
        When user presses Start:
        - Read mapping from UI
        - Optionally save to JSON
        - Notify controller to open main window
        """
        mapping = self.read_mapping_from_ui()
        if self.chk_save and self.chk_save.isChecked():
            save_mapping(mapping)
        self.mappingChosen.emit(mapping)

    def eventFilter(self, obj, event):
        """
        Capture key press into the focused QLineEdit.
        Mouse click starts capture mode, then next key press is stored.
        """
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
                # Start capturing key for this field
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
        """
        Convert Qt key event to a simple string
        (letters, digits, arrows, space, enter, etc.).
        """
        key = event.key()
        if key == QtCore.Qt.Key_Escape:
            # Use ESC to set "none"
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
        """Emit closed signal so controller can quit if needed."""
        self.closed.emit()
        return super().closeEvent(e)


# =============================================================
#                   MAIN WINDOW (VIDEO)
# =============================================================

class MainWindow(QtWidgets.QMainWindow):
    """
    Main camera window:
    - Shows video + HUD
    - Provides Setting / Close buttons
    - Keeps itself on top (for playing games)
    """
    requestSettings = QtCore.pyqtSignal()

    def __init__(self, mapping: dict):
        super().__init__()
        uic.loadUi(os.path.join(BASE_DIR, "pose.ui"), self)

        self.video_label = self.findChild(QtWidgets.QLabel, "videoLabel")

        self.btn_setting = self.findChild(QtWidgets.QPushButton, "Setting_Button")
        self.btn_close = self.findChild(QtWidgets.QPushButton, "Close_Button")

        self.btn_setting.clicked.connect(self.on_setting_clicked)
        self.btn_close.clicked.connect(self.close)

        # Keep window always on top
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)

        self.mapping = mapping
        self.worker = PoseWorker(mapping)
        self.worker.frameReady.connect(self.on_frame)
        self.worker.status.connect(self.on_status)
        self.worker.start()

        # Show bigger window and video area on startup
        self.show()
        self.resize(960, 720)
        self.video_label.setMinimumSize(640, 480)

        # Center this window on screen after layout is done
        QtCore.QTimer.singleShot(0, self.center_on_screen)

    def center_on_screen(self):
        """Move this window to the center of the primary screen."""
        screen = QtWidgets.QApplication.primaryScreen()
        if not screen:
            return
        screen_geo = screen.availableGeometry()
        geo = self.frameGeometry()
        geo.moveCenter(screen_geo.center())
        self.move(geo.topLeft())

    def on_setting_clicked(self):
        """Ask controller to show settings window again."""
        self.requestSettings.emit()

    @QtCore.pyqtSlot(QtGui.QImage)
    def on_frame(self, qimg):
        """
        Receive QImage from worker and draw it scaled into video_label.
        """
        pix = QtGui.QPixmap.fromImage(qimg)
        pix = pix.scaled(
            self.video_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(pix)

    @QtCore.pyqtSlot(str, str)
    def on_status(self, g, a):
        """
        Status callback (currently unused).
        Could be used to show text in status bar or logs.
        g: smoothed primary gesture
        a: joined key string (e.g., "right+shift")
        """
        pass

    def closeEvent(self, e):
        """
        Stop worker thread when main window is closed.
        """
        if self.worker:
            self.worker.stop()
            self.worker.wait(500)
        return super().closeEvent(e)


# =============================================================
#                 APP CONTROLLER (SWITCHING)
# =============================================================

class AppController(QtCore.QObject):
    """
    Controller to manage transitions between:
    - SettingsWindow (first)
    - MainWindow (after Start)
    """
    def __init__(self):
        super().__init__()
        self.settings = SettingsWindow()
        self.settings.mappingChosen.connect(self.on_mapping_chosen)
        self.settings.closed.connect(self.on_app_quit)
        self.settings.show()
        self.main = None

    def on_mapping_chosen(self, mapping):
        """
        Called when user presses Start in settings window.
        Show main window and apply mapping.
        """
        if self.main is None:
            self.main = MainWindow(mapping)
            self.main.requestSettings.connect(self.on_request_settings)
        else:
            self.main.mapping = mapping
            self.main.worker.set_mapping(mapping)
        self.main.show()
        self.settings.hide()

    def on_request_settings(self):
        """
        Called when user presses Setting in main window.
        Show settings window with current mapping.
        """
        if self.main:
            m = self.main.mapping
            self.settings.set_mapping(m)
        self.settings.show()

    def on_app_quit(self):
        """
        If settings window is closed without starting,
        quit the whole application.
        """
        QtWidgets.QApplication.quit()


# =============================================================
#                        ENTRY POINT
# =============================================================

def main():
    app = QtWidgets.QApplication(sys.argv)
    controller = AppController()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
