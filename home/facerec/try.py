# import mysql.connector
# i="Dhairya"
#
# def taskF(i):
#     mydb=mysql.connector.connect(host="localhost",user="root",password="", database="isms")
#     mycursor=mydb.cursor()
#     mycursor.execute("select * from home_face")
#     # print(name)
#     for name in mycursor:
#         name=list(name)
#         if name[1]==i:
#             print(name[1])
#             print(name[2])
#             print(name[3])
#             print(name[4])
#             print(name[5])
#             print(name[6])

#
# taskF(i)
import cv2
# from PIL import Image
# img_name = Image.open("C:/Users/Dhairya/PycharmProjects/Final/venv/ISMS/home/facerec/dataset/Dhairya/opencv_frame_0.png")
# cv2.imshow('Img', img_name)
img = cv2.imread("ISMS/static/dataset/Dhairya/opencv_frame_0.png", cv2.IMREAD_COLOR)
cv2.imshow("Img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()