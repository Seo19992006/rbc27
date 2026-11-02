import cv2
import numpy as np

# Thay đường dẫn bên dưới bằng đường dẫn thực tế tới video trên máy tính của bạn
video_path = r'C:\Users\Admin\Downloads\rbc27\thucliphop3d.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Không thể mở video. Hãy kiểm tra lại đường dẫn file!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        # Tự động quay lại đầu video khi chạy hết
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # 1. Chuyển sang hệ màu HSV để dễ lọc màu vật thể
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. Lọc màu Đỏ (Red) của khối hộp
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    # 3. Tìm đường viền (Contours) và khoanh vùng
    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 800:  # Lọc bỏ các điểm nhiễu nhỏ
            x, y, w, h = cv2.boundingRect(cnt)
            # Vẽ hình chữ nhật màu xanh lá quanh hộp đỏ
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Red Box', (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 4. Hiển thị cửa sổ xem kết quả
    cv2.imshow("Test Detection Windows", frame)

    # Bấm phím 'q' trên bàn phím để thoát chương trình
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()