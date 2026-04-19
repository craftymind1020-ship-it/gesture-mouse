import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# ---------------- SETUP ----------------
# Enable fail-safe (move mouse to corner to stop program)
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0  # remove delay for faster response

# Get screen resolution
screen_w, screen_h = pyautogui.size()

# Initialize MediaPipe hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

# Start webcam
cap = cv2.VideoCapture(0)

# ---------------- VARIABLES ----------------
# Cursor smoothing variables (reduces jitter)
cursor_x, cursor_y = pyautogui.position()
smooth = 0.3
margin = 5  # prevent cursor going off screen edges

# Click / Drag control
dragging = False
pinch_start_time = 0
click_threshold = 0.2  # short pinch = click, long pinch = drag

# Right click control
right_pinch = False
last_right_click = 0
click_delay = 0.4  # prevents multiple rapid right clicks

# Swipe gesture (copy/paste)
prev_swipe_x = None
swipe_mode = False
swipe_frames = 0
last_action_time = 0
action_delay = 1.0  # delay between copy/paste actions

# ---------------- MAIN LOOP ----------------
while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert to RGB (required for MediaPipe)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark

        # -------- CURSOR MOVEMENT --------
        # Using palm (landmark 9) for stable tracking
        palm_x = int(lm[9].x * w)
        palm_y = int(lm[9].y * h)

        # Map camera coordinates → screen coordinates
        screen_x = np.interp(palm_x, (80, w-80), (0, screen_w))
        screen_y = np.interp(palm_y, (80, h-80), (0, screen_h))

        # Smooth movement
        cursor_x += (screen_x - cursor_x) * smooth
        cursor_y += (screen_y - cursor_y) * smooth

        # Keep cursor inside screen
        cursor_x = np.clip(cursor_x, margin, screen_w - margin)
        cursor_y = np.clip(cursor_y, margin, screen_h - margin)

        pyautogui.moveTo(cursor_x, cursor_y)

        # -------- FINGER STATES --------
        # Detect which fingers are open (used for gestures)
        index_open  = lm[8].y < lm[6].y
        middle_open = lm[12].y < lm[10].y
        ring_open   = lm[16].y < lm[14].y
        pinky_open  = lm[20].y < lm[18].y

        current_time = time.time()

        # -------- PINCH (LEFT CLICK / DRAG) --------
        tx, ty = int(lm[4].x*w), int(lm[4].y*h)  # thumb
        ix, iy = int(lm[8].x*w), int(lm[8].y*h)  # index

        pinch_dist = np.hypot(tx - ix, ty - iy)

        if pinch_dist < 35:
            if pinch_start_time == 0:
                pinch_start_time = current_time
            elif current_time - pinch_start_time > click_threshold:
                # Long pinch → drag
                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True
        else:
            if dragging:
                pyautogui.mouseUp()
                dragging = False
            elif pinch_start_time != 0 and current_time - pinch_start_time < click_threshold:
                # Short pinch → click
                pyautogui.click()
            pinch_start_time = 0

        # -------- RIGHT CLICK --------
        px, py = int(lm[20].x*w), int(lm[20].y*h)  # pinky
        right_dist = np.hypot(tx - px, ty - py)

        if right_dist < 35 and not right_pinch:
            if current_time - last_right_click > click_delay:
                pyautogui.rightClick()
                last_right_click = current_time
                right_pinch = True
        elif right_dist > 60:
            right_pinch = False

        # -------- SWIPE MODE (2 FINGERS ONLY) --------
        two_fingers = index_open and middle_open and not ring_open and not pinky_open

        if two_fingers:
            swipe_frames += 1
        else:
            swipe_frames = 0
            swipe_mode = False
            prev_swipe_x = None

        if swipe_frames > 3:
            swipe_mode = True

        # -------- SWIPE ACTION (COPY / PASTE) --------
        if swipe_mode:
            finger_x = int(lm[8].x * w)

            if prev_swipe_x is not None:
                dx = finger_x - prev_swipe_x

                if abs(dx) > 40 and current_time - last_action_time > action_delay:
                    if dx > 0:
                        pyautogui.hotkey('ctrl', 'c')  # copy
                        cv2.putText(frame, "COPY", (20,150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    else:
                        pyautogui.hotkey('ctrl', 'v')  # paste
                        cv2.putText(frame, "PASTE", (20,150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                    last_action_time = current_time

            prev_swipe_x = finger_x

        # -------- VISUAL FEEDBACK --------
        # Draw circle on palm for tracking visualization
        cv2.circle(frame, (palm_x, palm_y), 8, (255,0,255), -1)

    # Show camera window
    cv2.imshow("Gesture Mouse FINAL PRO", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()