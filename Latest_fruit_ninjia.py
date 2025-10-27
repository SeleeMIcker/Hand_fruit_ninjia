import cv2
import mediapipe as mp
import numpy as np
import random
import time
from collections import deque
from playsound import playsound
import threading
import os

# ================= Hand Tracking Setup =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ================= Game Setup =================
cap = cv2.VideoCapture(0)
WIDTH, HEIGHT = 1280, 720

FRUIT_RADIUS = 50
FRUIT_SPEED_Y = -25
GRAVITY = 0.5
SPAWN_INTERVAL = 10

FRUIT_COLORS = [(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,0,255),(255,165,0)]
fruits = []
score = 0
final_score = 0

# Knife trail
trail_length = 15
knife_trail = deque(maxlen=trail_length)
frame_count = 0

# Game state
game_started = False
game_over = False
target_score = 1000
time_limit = 30  # seconds
start_time = None

# Slice sound file
ASSETS_DIR = os.path.join(os.path.dirname(__file__),"assets")
sound_file = os.path.join(os.path.dirname(__file__), 'slice.wav')

def play_sound_nonblocking(sound_file):
    if os.path.exists(sound_file):
        threading.Thread(target=playsound, args=(sound_file,), daemon=True).start()

# ================= Helper Classes =================
class Fruit:
    def __init__(self):
        self.x = random.randint(100, WIDTH-100)
        self.y = HEIGHT + FRUIT_RADIUS
        self.vx = random.uniform(-4, 4)
        self.vy = FRUIT_SPEED_Y
        self.radius = FRUIT_RADIUS
        self.color = random.choice(FRUIT_COLORS)
        self.cutted = False
        self.flash_timer = 0
        self.score_popup_timer = 0
        self.split_fruits = []
        self.already_counted = False
        self.sliceable = True

    def move(self):
        self.vy += GRAVITY
        self.x += self.vx
        self.y += self.vy

    def split(self):
        left = Fruit()
        right = Fruit()
        left.x, left.y = self.x, self.y
        right.x, right.y = self.x, self.y
        left.vx, left.vy = -3, -random.uniform(5,8)
        right.vx, right.vy = 3, -random.uniform(5,8)
        left.radius = self.radius // 2
        right.radius = self.radius // 2
        left.color = self.color
        right.color = self.color
        left.sliceable = False
        right.sliceable = False
        self.split_fruits = [left, right]

# ================= Main Game Loop =================
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    index_tip = None
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        index_tip = hand_landmarks.landmark[8]
        ix, iy = int(index_tip.x * WIDTH), int(index_tip.y * HEIGHT)
        knife_trail.append((ix, iy))
    else:
        knife_trail.clear()

    # ================= Start Screen =================
    if not game_started:
        # Draw start button
        start_btn_x1, start_btn_y1 = WIDTH//2 - 200, HEIGHT//2 - 80
        start_btn_x2, start_btn_y2 = WIDTH//2 + 200, HEIGHT//2 + 80
        cv2.rectangle(frame, (start_btn_x1, start_btn_y1), (start_btn_x2, start_btn_y2), (0,255,0), -1)
        cv2.putText(frame, "START", (WIDTH//2 - 100, HEIGHT//2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 5)

        # Check if hand taps inside button
        if index_tip:
            if start_btn_x1 < ix < start_btn_x2 and start_btn_y1 < iy < start_btn_y2:
                cv2.putText(frame, "Starting...", (WIDTH//2 - 150, HEIGHT//2 + 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)
                game_started = True
                final_score = score
                score = 0
                fruits = []
                start_time = time.time()
                continue  # skip rest this frame

        cv2.imshow("Hand Fruit Ninja", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    # ================= Game Running =================
    elapsed_time = time.time() - start_time
    remaining_time = max(0, int(time_limit - elapsed_time))

    if remaining_time <= 0 or score >= target_score:
        game_over = True
        game_started = False
        continue

    frame_count += 1
    if frame_count % SPAWN_INTERVAL == 0:
        fruits.append(Fruit())

    new_fruits = []
    for fruit in fruits:
        fruit.move()
        if index_tip and not fruit.cutted and not fruit.already_counted and fruit.sliceable:
            if (fruit.x - ix)**2 + (fruit.y - iy)**2 < fruit.radius**2:
                fruit.cutted = True
                fruit.flash_timer = 5
                fruit.score_popup_timer = 20
                fruit.split()
                fruit.already_counted = True
                score += 1
                play_sound_nonblocking(sound_file)

        if fruit.cutted:
            if fruit.flash_timer > 0:
                cv2.circle(frame, (int(fruit.x), int(fruit.y)), int(fruit.radius), (255,255,255), -1)
                fruit.flash_timer -= 1
            for half in fruit.split_fruits:
                half.move()
                cv2.circle(frame, (int(half.x), int(half.y)), int(half.radius), half.color, -1)
                new_fruits.append(half)
            if fruit.score_popup_timer > 0:
                cv2.putText(frame, "+1", (int(fruit.x), int(fruit.y - fruit.radius)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
                fruit.score_popup_timer -= 1
        else:
            cv2.circle(frame, (int(fruit.x), int(fruit.y)), int(fruit.radius), fruit.color, -1)
            new_fruits.append(fruit)

    fruits = [f for f in new_fruits if f.y - f.radius < HEIGHT + 50]

    # Knife trail
    for i in range(1, len(knife_trail)):
        cv2.line(frame, knife_trail[i-1], knife_trail[i], (255,255,255), 3)


    # Score & Timer
    cv2.putText(frame, f"Score: {score}", (50,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
    cv2.putText(frame, f"Time: {remaining_time}s", (WIDTH-300, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 3)
    cv2.putText(frame, f"Target: {target_score}", (WIDTH//2 - 120, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)

    cv2.imshow("Hand Fruit Ninja", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # ================= Game Over Screen =================
    if game_over:
        frame[:] = (0, 0, 0)
        if score >= target_score:
            msg = "GAME COMPLETE!"
        else:
            msg = "TIME'S UP!"
            
        cv2.putText(frame, msg, (WIDTH//2 - 350, HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 5)
        cv2.putText(frame, f"Your Score: {final_score}", (WIDTH//2 - 250, HEIGHT//2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 4)
        cv2.putText(frame, "Show hand to return to START", (WIDTH//2 - 350, HEIGHT//2 + 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3)

        cv2.imshow("Hand Fruit Ninja", frame)
        cv2.waitKey(3000)
        game_over = False
        game_started = False

cap.release()
cv2.destroyAllWindows()
