from django.shortcuts import render, HttpResponse, redirect
from django.http import HttpResponseRedirect
from django.http.response import StreamingHttpResponse
from django.urls import reverse
from home.models import Face, Entry, register, login, Save, accuracy, cat, LastFace, super
from home.facerec.train_faces import train
import pickle
import os
import schedule
import time
from home.facerec.identify_faces import VideoCamera1
from home.facerec.identify_facesOut import VideoCamera1Out
import mysql.connector
import face_recognition
from home.facerec.train_faces import trainer
from cachetools import TTLCache
import collections
import cv2
import pytesseract
import datetime
import numpy as np
import imutils
from django.contrib import messages
import threading

def Login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        task=register.objects.all()
        for i in task:
            if i.username==username and i.password==password:
                print(username)
                print(password)
                datein = datetime.datetime.now().date()
                timein = datetime.datetime.now().time()
                log=login(username=username,password=password,datein=datein,timein=timein)
                log.save()
                request.session["username"] = username
                return render(request, 'index.html',{'username':username})
            else:
                pass
                # messages.warning(request, 'No such Username is available!!')
                # return render(request, 'Login.html')

    return render(request, 'Login.html')

def Register(request):
    if request.method=="POST":
        username=request.POST.get('username')
        password = request.POST.get('password')
        u = register.objects.all()
        for i in u:
            if i.username == username:
                messages.warning(request,'This username is not available!!')
                return render(request, 'Register.html')
            else:
                pass
        date = datetime.datetime.now().date()
        time = datetime.datetime.now().time()
        reg=register(username=username,password=password, date=date, time=time)
        reg.save()
        messages.success(request,'Congratulations! You have been successfully registered!')
    return render(request,'Register.html')

def logout(request):
    if request.session['username'] != None:
        try:
            username = request.session['username']
            t = login.objects.filter(username=username).last()
            t.dateout = datetime.datetime.now().date()
            t.timeout = datetime.datetime.now().time()
            t.save()
            del request.session['username']
            return redirect('../Login/')
        except:
            pass
    return redirect('../Login/')

def index(request):
    return render(request, "index.html")

def section(request, num):
    if 1 <= num <= 5:
        if num==1:
            return HttpResponseRedirect('view')
        if num==2:
            return HttpResponseRedirect('add')
        if num==3:
            return HttpResponseRedirect('checklog')
        if num==4:
            return HttpResponseRedirect('category')
    else:
        raise Http404("No such section")


class VideoCamera1(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()
    # # def get_frame(self):
    #     success, img=self.video.read()

    def identify_faces(self):
        buf_length = 10
        known_conf = 5
        buf = [[]] * buf_length
        i = 0
        cache = TTLCache(maxsize=20, ttl=10)

        def identify1(frame, name, buf, buf_length, known_conf):
            count=0
            if name in cache:
                return
                count = 0
            for ele in buf:
                count += ele.count(name)

            if count >= known_conf:
                timestamp = datetime.datetime.now(tz=timezone.utc)
                print(name, timestamp)
                cache[name] = 'detected'
                path = 'static/detected/{}_{}.jpg'.format(name, timestamp)
                cv2.imwrite(path, frame)
                try:
                    emp = Employee.objects.get(name=name)
                    emp.detected_set.create(time_stamp=timestamp)
                except:
                    pass

        def predict(rgb_frame, knn_clf=None, model_path=None, distance_threshold=0.3):
            if knn_clf is None and model_path is None:
                raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

            # Load a trained KNN model (if one was passed in)
            if knn_clf is None:
                with open(model_path, 'rb') as f:
                    knn_clf = pickle.load(f)

                # Load image file and find face locations
                # X_img = face_recognition.load_image_file(X_img_path)
            X_face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=2)

            # If no faces are found in the image, return an empty result.
            if len(X_face_locations) == 0:
                return []

                # Find encodings for faces in the test image
            faces_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=X_face_locations)

            # Use the KNN model to find the best matches for the test face
            closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=5)
            are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
            # print(closest_distances)
            # Predict classes and remove classifications that aren't within the threshold
            return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
                    zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

        process_this_frame = True

        while True:
            # Grab a single frame of video
            ret, frame = self.video.read()

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_frame = small_frame[:, :, ::-1]

            if process_this_frame:
                predictions = predict(rgb_frame, model_path="static/models/trained_model.clf")
                # print(predictions)

            process_this_frame = not process_this_frame

            face_names = []

            for name, (top, right, bottom, left) in predictions:
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                # font = cv2.FONT_HERSHEY_DUPLEX
                if name!='unknown':
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (0, 0, 0), 1)
                    identify1(frame, name, buf, buf_length, known_conf)
                    file = open("name.txt", 'w')
                    file.write(str(name))
                    file.close()
                else:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (0, 0, 0), 1)
                    identify1(frame, name, buf, buf_length, known_conf)

                face_names.append(name)

                # Name.append(name)

            #
            # buf[i] = face_names
            # i = (i + 1) % buf_length
            #
            # # print(buf)
            #
            # # Display the resulting image
            # # cv2.imshow('Video', frame)
            resize = cv2.resize(frame, (300, 360), interpolation=cv2.INTER_LINEAR)
            # frame_flip = cv2.flip(resize, 1)
            ret, jpeg = cv2.imencode('.jpg', resize)
            return jpeg.tobytes()

def gen(identify_faces):
    while True:
        frame=identify_faces.identify_faces()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_feed1(request):
    #cam = VideoCamera1()
    return StreamingHttpResponse(gen(VideoCamera1()), content_type="multipart/x-mixed-replace;boundary=frame")

def click(dirName):
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

    cam=cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()

        cv2.imshow("Video", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

        elif k % 256 == 32:
            # SPACE pressed
            img_name = f"static/dataset/{dirName}/opencv_frame_{img_counter}.png"
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()

def view(request):
    sup = Face.objects.filter(cat='Supervisor')
    f = open('name.txt', 'r')
    name = str(f.readline())
    task = Face.objects.all()
    for i in task:
        # print(i.name)
        if i.name == name:
            # i.name=(i.name).replace(' ','_')
            img_name = f"static/dataset/{i.name}/opencv_frame_0.png"
            print(i.name)
            Name = i.name
            print(i.rank)
            Rank = i.rank
            print(i.number)
            Number = i.number
            print(i.adharno)
            Adhar = i.adharno
            print(i.cat)
            Cat = i.cat
            gender = i.gender
            print(gender)
            print(i.blacklist)
            B = i.blacklist
            print(i.snumber)
            snumber = i.snumber
            if 'correct' in request.POST:
                a = accuracy(unknown=0, correct=1, incorrect=0, name=Name)
                a.save()
                return HttpResponseRedirect('/')

            if 'incorrect' in request.POST:
                a = accuracy(unknown=0, correct=0, incorrect=1, name=Name)
                a.save()
                return HttpResponseRedirect('/')

            if 'unknown' in request.POST:
                a = accuracy(unknown=1, correct=0, incorrect=0, name=Name)
                a.save()
                return HttpResponseRedirect('/')

            if 'In' in request.POST:
                supervisor = request.POST.get('supervisor')
                token = request.POST.get('token')
                date = datetime.datetime.now().date()
                time = datetime.datetime.now().time()

                datein = datetime.datetime.now().date()
                timein = datetime.datetime.now().time()
                a = Save(name=Name, rank=Rank, number=Number, adharno=Adhar, snumber=snumber, cat=Cat,
                         supervisor=supervisor, token=token, blacklist=B, timein=timein, datein=datein)
                a.save()
                return HttpResponseRedirect('/')

            if 'Out' in request.POST:
                a = Save.objects.filter(name=Name).last()
                a.timeout = datetime.datetime.now().time()
                a.dateout = datetime.datetime.now().date()
                timein = a.timein
                datein = a.datein
                timeout = a.timeout
                dateout = a.dateout
                a.save()
                return HttpResponseRedirect('/')
            else:
                pass
            return render(request, 'view.html',
                          {'Name': Name, 'Number': Number, 'Rank': Rank, 'Adhar': Adhar, 'snumber': snumber, 'Cat': Cat,
                           'B': B, 'img_name': img_name, 'gender': gender, 'sup':sup})

    return render(request, 'view.html')


def category(request):
    if request.method == "POST":
        category = request.POST.get('category')
        date = datetime.datetime.now().date()
        time = datetime.datetime.now().time()
        c = cat(category=category, date=date, time=time)
        c.save()
        return HttpResponseRedirect('/')
    else:
        pass
    return render(request,'category.html')

def add(request):
    if request.method == "POST":
        name = request.POST.get('name')
        # print(name)
        # name = name.replace(' ', '_')
        rank = request.POST.get('rank')
        number = request.POST.get('number')
        adharno = request.POST.get('adharno')
        blacklist = request.POST.get('blacklist')
        Cat = request.POST.get('cat')
        gender = request.POST.get('gender')
        snumber = request.POST.get('snumber')
        date = datetime.datetime.now().date()
        time = datetime.datetime.now().time()
        username='Dhairya'
        Add1 = Face(name=name, rank=rank, number=number, adharno=adharno, blacklist=blacklist, cat=Cat, gender=gender,
                    snumber=snumber, username=username, date=date, time=time)
        Add1.save()
        dirName = name
        click(dirName)
        return HttpResponseRedirect('/')
    else:
        pass
    c = cat.objects.all()
    return render(request,'add.html',{'c':c})

def checklog(request):
    task = Save.objects.all()
    return render(request,'checklog.html',{'task': task})

