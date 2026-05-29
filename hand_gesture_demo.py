"""
AR手语数字人项目 - 完整版
功能：
  - 数字0-9、打电话、嘘等手势识别
  - 睁眼/闭眼、张嘴/闭嘴检测
  - 左右手识别
依赖：pip install mediapipe opencv-python numpy Pillow
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from PIL import Image, ImageDraw, ImageFont

#  初始化
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 手势名称映射
GESTURE_NAMES = {
    "zero": "数字0", "one": "数字1", "two": "数字2",
    "three": "数字3", "four": "数字4", "five": "数字5",
    "six": "数字6", "seven": "数字7", "eight": "数字8",
    "nine": "数字9", "call": "打电话", "shush": "嘘",
    "fist": "拳头", "open": "张开", "unknown": "识别中..."
}

#  人脸关键点索引
LEFT_EYE_TOP, LEFT_EYE_BOTTOM = 159, 145
RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM = 386, 374
UPPER_LIP, LOWER_LIP = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 61, 291

#  中文绘制
def draw_chinese_text(img, text, pos, font_size=20, color=(255,255,255)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = None
    for fp in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]:
        try:
            font = ImageFont.truetype(fp, font_size)
            break
        except:
            pass
    if font is None:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


#  手势识别
def get_finger_states(landmarks, handedness):
    """获取5根手指的伸展状态"""
    fingers = []
    
    # 拇指（区分左右手）
    if handedness == "Left":
        fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)
    else:
        fingers.append(1 if landmarks[4].x > landmarks[3].x else 0)
    
    # 其他四指
    fingers.append(1 if landmarks[8].y < landmarks[6].y else 0)
    fingers.append(1 if landmarks[12].y < landmarks[10].y else 0)
    fingers.append(1 if landmarks[16].y < landmarks[14].y else 0)
    fingers.append(1 if landmarks[20].y < landmarks[18].y else 0)
    
    return fingers


def recognize_gesture(landmarks, handedness):
    """识别手势"""
    fingers = get_finger_states(landmarks, handedness)
    thumb, index, middle, ring, pinky = fingers
    
    gesture = "unknown"
    
    # 数字0：握拳
    if sum(fingers) == 0:
        thumb_tip = landmarks[4]
        index_mcp = landmarks[5]
        dist = np.sqrt((thumb_tip.x - index_mcp.x)**2 + (thumb_tip.y - index_mcp.y)**2)
        gesture = "zero" if dist > 0.08 else "fist"
    
    # 数字1-5
    elif sum(fingers) == 1 and index == 1:
        gesture = "one"
    elif sum(fingers) == 2 and index == 1 and middle == 1:
        gesture = "two"
    elif sum(fingers) == 3 and index == 1 and middle == 1 and ring == 1:
        gesture = "three"
    elif sum(fingers) == 4 and thumb == 0:
        gesture = "four"
    elif sum(fingers) == 5:
        gesture = "five"
    
    # 数字6-9
    elif sum(fingers) == 1 and thumb == 1:
        gesture = "six"
    elif sum(fingers) == 2 and thumb == 1 and index == 1:
        gesture = "seven"
    elif sum(fingers) == 3 and thumb == 1 and index == 1 and middle == 1:
        gesture = "eight"
    elif sum(fingers) == 4 and thumb == 1:
        gesture = "nine"
    
    # 特殊手势
    elif thumb == 1 and pinky == 1 and index == 0 and middle == 0 and ring == 0:
        gesture = "call"
    elif sum(fingers) == 1 and pinky == 1:
        gesture = "shush"
    
    return gesture, GESTURE_NAMES.get(gesture, "识别中..."), fingers


#  脸部检测 =
def detect_face(face_landmarks, w, h):
    """检测睁眼/闭眼、张嘴/闭嘴"""
    lm = face_landmarks.landmark
    
    # 计算眼睛纵横比（EAR）
    def eye_ear(top, bottom, left, right):
        height = np.sqrt((lm[top].x - lm[bottom].x)**2 + (lm[top].y - lm[bottom].y)**2)
        width = np.sqrt((lm[left].x - lm[right].x)**2 + (lm[left].y - lm[right].y)**2)
        return height / (width + 1e-6)
    
    left_ear = eye_ear(LEFT_EYE_TOP, LEFT_EYE_BOTTOM, 33, 133)
    right_ear = eye_ear(RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, 362, 263)
    avg_ear = (left_ear + right_ear) / 2
    
    eye_state = "闭眼" if avg_ear < 0.18 else "睁眼"
    
    # 计算嘴巴纵横比（MAR）
    mouth_height = np.sqrt((lm[UPPER_LIP].x - lm[LOWER_LIP].x)**2 + (lm[UPPER_LIP].y - lm[LOWER_LIP].y)**2)
    mouth_width = np.sqrt((lm[MOUTH_LEFT].x - lm[MOUTH_RIGHT].x)**2 + (lm[MOUTH_LEFT].y - lm[MOUTH_RIGHT].y)**2)
    mar = mouth_height / (mouth_width + 1e-6)
    
    mouth_state = "张嘴" if mar > 0.3 else "闭嘴"
    
    return eye_state, mouth_state


# ==================== 绘图函数 ====================
def draw_panel(img, info, fps, history):
    """绘制信息面板"""
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (320, 320), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    y = 35
    img = draw_chinese_text(img, f"FPS: {fps:.1f}", (20, y), 22, (0, 255, 255))
    y += 40
    
    # 手部信息
    if info.get("gesture"):
        hand_label = info.get("hand", "手")
        img = draw_chinese_text(img, f"{hand_label}: {info['gesture']}", (20, y), 26, (0, 255, 0))
        y += 40
    
    # 手指状态
    if info.get("fingers"):
        f = info["fingers"]
        names = ["拇", "食", "中", "无", "小"]
        status = " ".join([f"{n}{'●' if s else '○'}" for n, s in zip(names, f)])
        img = draw_chinese_text(img, status, (20, y), 18, (200, 200, 200))
        y += 35
    
    # 脸部信息
    if info.get("eye"):
        img = draw_chinese_text(img, f"眼睛: {info['eye']}", (20, y), 20, (255, 255, 0))
        y += 32
    if info.get("mouth"):
        img = draw_chinese_text(img, f"嘴巴: {info['mouth']}", (20, y), 20, (255, 200, 255))
        y += 35
    
    # 历史
    if history:
        recent = " → ".join(list(history)[-4:])
        img = draw_chinese_text(img, f"历史: {recent}", (20, y), 16, (150, 150, 150))
    
    return img


def draw_guide(img):
    """手势指南"""
    h, w = img.shape[:2]
    x, y = w - 180, h - 200
    
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (w-10, h-10), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    guides = ["数字: 0-9", "打电话", "嘘", "睁眼/闭眼", "张嘴/闭嘴", "左右手"]
    yy = y + 25
    img = draw_chinese_text(img, "功能", (x+15, yy), 18, (255, 255, 255))
    yy += 28
    for g in guides:
        img = draw_chinese_text(img, f"• {g}", (x+15, yy), 14, (180, 180, 180))
        yy += 22
    return img


# ==================== 主程序 ====================
def main():
    print("=" * 50)
    print("  手势识别完整版")
    print("  数字0-9 + 打电话 + 嘘 + 睁眼闭眼 + 张嘴闭嘴 + 左右手")
    print("=" * 50)
    print("\n操作: 'q'退出 | 'r'清空历史")
    print()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[错误] 无法打开摄像头")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = time.time()
    fps = 0
    history = deque(maxlen=20)
    last_gesture = None
    hold_count = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    ) as hands, mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                continue

            img = cv2.flip(img, 1)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            hand_results = hands.process(rgb)
            face_results = face_mesh.process(rgb)

            # FPS
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - prev_time, 0.001)
            prev_time = now

            info = {}
            h, w = img.shape[:2]

            # ====== 手部处理 ======
            if hand_results.multi_hand_landmarks:
                for idx, hand_lms in enumerate(hand_results.multi_hand_landmarks):
                    # 绘制手部
                    mp_drawing.draw_landmarks(
                        img, hand_lms, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # 获取左右手（注意：画面翻转后左右手标签也要翻转）
                    raw_hand = hand_results.multi_handedness[idx].classification[0].label
                    display_hand = "右手" if raw_hand == "Left" else "左手"
                    
                    # 识别手势
                    key, name, fingers = recognize_gesture(hand_lms.landmark, raw_hand)
                    info["gesture"] = name
                    info["fingers"] = fingers
                    info["hand"] = display_hand

                    # 稳定记录
                    if key == last_gesture:
                        hold_count += 1
                        if hold_count == 10 and key != "unknown":
                            history.append(f"{display_hand}:{name}")
                    else:
                        last_gesture = key
                        hold_count = 0

                    # 绘制指尖
                    tip_ids = [4, 8, 12, 16, 20]
                    for i, tid in enumerate(tip_ids):
                        if fingers[i] == 1:
                            cx = int(hand_lms.landmark[tid].x * w)
                            cy = int(hand_lms.landmark[tid].y * h)
                            cv2.circle(img, (cx, cy), 8, (0, 255, 0), -1)
                    
                    # 在手旁标注左右手
                    wrist_x = int(hand_lms.landmark[0].x * w)
                    wrist_y = int(hand_lms.landmark[0].y * h)
                    img = draw_chinese_text(img, display_hand, (wrist_x - 30, wrist_y + 30), 22, (255, 100, 100))

            # ====== 脸部处理 ======
            if face_results.multi_face_landmarks:
                for face_lms in face_results.multi_face_landmarks:
                    # 绘制脸部关键点
                    for idx in [33, 133, 362, 263, 61, 291, 13, 14, 
                                LEFT_EYE_TOP, LEFT_EYE_BOTTOM, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM]:
                        x = int(face_lms.landmark[idx].x * w)
                        y = int(face_lms.landmark[idx].y * h)
                        cv2.circle(img, (x, y), 2, (255, 255, 0), -1)
                    
                    # 检测脸部状态
                    eye_state, mouth_state = detect_face(face_lms, w, h)
                    info["eye"] = eye_state
                    info["mouth"] = mouth_state

            # 绘制面板
            img = draw_panel(img, info, fps, history)
            img = draw_guide(img)

            cv2.imshow("Gesture + Face Detection", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"\n识别记录: {list(history)}")
                break
            elif key == ord('r'):
                history.clear()
                print("[清空] 历史已清空")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
