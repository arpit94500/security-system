import cv2
import os
import face_recognition

class Video(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def click(self,dirName):

        img_counter = 0

        # dirName = input("Please enter your name: ").upper()
        # dirID = input("Please enter ID: ")

        DIR = f"static/dataset/{dirName}"

        try:
            os.mkdir(DIR)
            print("Directory ", dirName, " Created ")
        except FileExistsError:
            print("Directory ", dirName, " already exists")
            img_counter = len(os.listdir(DIR))

        while True:
            ret, frame = self.video.read()
            # rgb_small_frame = frame[:, :, ::-1]

            # face_locations = face_recognition.face_locations(rgb_small_frame)
            # print(len(face_locations))

            cv2.imshow("Video", frame)
            if not ret:
                break
            k = cv2.waitKey(1)

            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break


            # SPACE pressed
            elif k % 256 == 32:
                img_name = f"static/dataset/{dirName}/opencv_frame_{img_counter}.png"
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
                img_counter += 1

        cam.release()

        cv2.destroyAllWindows()
