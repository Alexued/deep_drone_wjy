import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

ret, old_frame = cap.read()
if not ret:
    print("Can't read video")
    exit()

old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('Event.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

ratio_threshold = 2.0
# threshold = 30
frame_index = 0
# 获取开始时间
ticks = cv2.getTickCount()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame_gray = cv2.cuda_GpuMat(frame).cvtColor(cv2.COLOR_BGR2GRAY)

    diff = cv2.subtract(frame_gray.astype(int), old_frame.astype(int))
    increase = np.clip(diff, 0, 255).astype(np.uint8)
    decrease = np.clip(-diff, 0, 255).astype(np.uint8)

    # threshold可以调整，太小噪点多，太大事件检测不灵敏，现在不采用
    # diff = frame_gray - old_frame
    # frame_std = np.std(diff)
    # threshold = max(10, frame_std * 0.5)  # 保证最小值为 10
    # increase = (diff > threshold).astype(np.uint8) * 255
    # decrease = (diff < -threshold).astype(np.uint8) * 255

    threshold = 30
    increase_event = (increase > threshold).astype(np.uint8) * 255
    decrease_event = (decrease > threshold).astype(np.uint8) * 255

    # 动态计算字体大小和位置
    font_scale = frame.shape[1] / 800  # 基于窗口宽度动态调整字体大小
    thickness = max(1, int(frame.shape[1] / 800))  # 动态调整字体线条厚度
    text_position_event = (frame.shape[1] - int(200 * font_scale), int(30 * font_scale))  # 事件文字位置
    text_position_fps = (int(10 * font_scale), int(30 * font_scale))  # FPS文字位置

    event_img = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    event_img[:, :, 2] = increase_event
    event_img[:, :, 0] = decrease_event

    increase_count = np.sum(increase_event) / 255.0
    decrease_count = np.sum(decrease_event) / 255.0

    if increase_count > decrease_count * ratio_threshold:
        cv2.putText(event_img, '++!Event!++', text_position_event, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
    elif decrease_count > increase_count * ratio_threshold:
        cv2.putText(event_img, '--!Event!--', text_position_event, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)

    # 计算帧率
    current_ticks = cv2.getTickCount()
    time_interval = (current_ticks - ticks) / cv2.getTickFrequency()
    ticks = current_ticks
    fps = 1.0 / time_interval if time_interval > 0 else 0
    cv2.putText(event_img, f"FPS: {fps:.2f}", text_position_fps, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)


    if frame_index % 2 == 0:
        cv2.imshow("Event Camera, Press 'q' to quit", event_img)
    # cv2.imshow("Event Camera, Press 'q' to quit", event_img)
    out.write(event_img)
    
    old_frame = frame_gray

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()