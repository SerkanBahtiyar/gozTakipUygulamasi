import cv2

cap = cv2.VideoCapture(".venv/c.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (800, 600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # "gri" -> "gray" düzeltildi
    blur = cv2.GaussianBlur(gray, (7, 7), 0)  # "gri" -> "gray" düzeltildi
    _, thresh = cv2.threshold(blur, 10, 200, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.line(frame, (x + int(w / 2), 0), (x + int(h / 2), frame.shape[0]), (0, 255, 0), 2)
        cv2.line(frame, (0, y + int(h / 2)), (frame.shape[1], int(h / 2)), (0, 255, 0), 2)

    cv2.imshow("goz", frame)
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
