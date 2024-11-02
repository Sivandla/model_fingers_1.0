import cv2
import mediapipe as mp
import os



mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Mở webcam
cap = cv2.VideoCapture(0)

# Tạo thư mục lưu trữ dữ liệu
data_dir = 'D:\\Code\\Python\\bot\\hand01\\sign_language_data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

label = input("Nhập ký hiệu bạn muốn thu thập: ")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Xử lý hình ảnh để nhận diện bàn tay
    results = hands.process(image)

    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Lưu hình ảnh
            img_name = os.path.join(data_dir, f"{label}_{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}.jpg")
            cv2.imwrite(img_name, frame)

    # Hiển thị hình ảnh
    cv2.imshow('Collect Data', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
