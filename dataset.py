import cv2
import os
from facenet_pytorch import MTCNN

vid = "shanti.mp4"

def create_dataset(vid):
    video = cv2.VideoCapture(vid)
    mtcnn = MTCNN()
    vid_name = os.path.basename(vid)
    user_name = os.path.splitext(vid_name)[0]
    os.mkdir(f"dataset/train/{user_name}")
    count = 0
    while True:
        _, img = video.read()
        
        boxes, _ = mtcnn.detect(img)

        if boxes is not None:
            for i, (x, y, w, h) in enumerate(boxes):
                x, y, w, h = int(x), int(y), int(w), int(h)
                face_img = img[y:y+h, x:x+w]
              
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 225, 0), 2)
                count += 1
                cv2.imwrite(f"dataset/train/{user_name}/User.{count}.jpg", face_img)
                cv2.imshow('image', img)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break
        elif count >= 1000:  # Take 200 face samples and stop video
            break

    print("\n[INFO] Exiting Program and cleaning up")
    video.release()
    cv2.destroyAllWindows()

create_dataset(vid)
