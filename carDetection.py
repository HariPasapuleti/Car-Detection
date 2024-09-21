import cv2

cascade_source = 'cars.xml'
video_source = 'videos/cars.mp4'

cap = cv2.VideoCapture(video_source)
car_cascade = cv2.CascadeClassifier(cascade_source)

# Get the frame width and height
frame_width = 600
frame_height = 400

# ret, frame = cap.read()
# frame = cv2.resize(frame, (600, 400))
# if ret:
#     cv2.imshow("Cars Show ", frame) 
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))



while True:
    ret, img = cap.read()

    if(type(img) == type(None)):
        print("End of video file or error reading the frame.")
        break

    img = cv2.resize(img, (frame_width, frame_height))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray_img = cv2.equalizeHist(gray_img)

    cars = car_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for(x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Cars Detection', img)
    out.write(img)

    if cv2.waitKey(1) == ord('n'):
        print("Video processing interrupted by user.")
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing completed and saved as output.mp4")
