import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime, timedelta
import time
import tkinter as tk
import threading
import csv

def main():
    root = tk.Tk()

    window_width = 1000
    window_height = 1000
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_coordinate = (screen_width - window_width) // 2
    y_coordinate = (screen_height - window_height) // 2

    root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

    mainFrame = tk.Frame(root, padx=10, pady=10)
    mainFrame.place(relx=0.5, rely=0.5, anchor="center")

    label = tk.Label(mainFrame, text="Face Recognition Attendance", font=("Helvetica", 25))
    label.pack()

    # Create a label to display success messages
    global success_label
    success_label = tk.Label(mainFrame, text="", font=("Helvetica", 18))
    success_label.pack()

    addNewEmployeeFrame = tk.Frame(mainFrame, pady=20, padx=20)
    addNewEmployeeFrame.pack()

    addNewEmployeeBtn = tk.Button(addNewEmployeeFrame, text="ADD NEW EMPLOYEE", command=showRegistration, padx=50, pady=10)
    addNewEmployeeBtn.pack()

    timeInFrame = tk.Frame(mainFrame, padx=20, pady=20)
    timeInFrame.pack()

    timeInBtn = tk.Button(timeInFrame, text="TIME IN BY FACE BIO", command=lambda: detectFace("time in"), padx=50, pady=10)
    timeInBtn.pack()

    timeOutFrame = tk.Frame(mainFrame, padx=20, pady=20)
    timeOutFrame.pack()

    timeOutBtn = tk.Button(timeOutFrame, text="TIME OUT BY FACE BIO", command=lambda: detectFace("time out"), padx=50, pady=10)
    timeOutBtn.pack()

    csv_file_path = 'Attendance.csv'
    employee_totals = calculateTotalHours(csv_file_path)
    for employee, total_hours in employee_totals.items():
        print(f"Total hours worked by {employee}: {total_hours}")

    root.mainloop()

def calculateTotalHours(csv_file_path):
    employee_totals = {}
    with open(csv_file_path, 'r') as f:
        myDataList = f.readlines()
        for line in myDataList:
            entry = line.strip().split(',')
            if len(entry) >= 4 and entry[3] == 'TIME IN':
                employee_name = entry[0]
                time_in = datetime.strptime(entry[1], '%I:%M:%S:%p')
                for line_out in myDataList:
                    entry_out = line_out.strip().split(',')
                    if entry_out[0] == employee_name and entry_out[2] == entry[2] and entry_out[3] == 'TIME OUT':
                        time_out = datetime.strptime(entry_out[1], '%I:%M:%S:%p')
                        total_hours = time_out - time_in
                        if employee_name in employee_totals:
                            employee_totals[employee_name] += total_hours
                        else:
                            employee_totals[employee_name] = total_hours
                        break
    return employee_totals


def showRegistration():
    root = tk.Tk()

    root.title("ADD NEW EMPLOYEE")

    window_width = 300
    window_height = 300
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_coordinate = (screen_width - window_width) // 2
    y_coordinate = (screen_height - window_height) // 2

    root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

    mainFrame = tk.Frame(root, padx=10, pady=10)
    mainFrame.place(relx=0.5, rely=0.5, anchor="center")

    nameLabel = tk.Label(mainFrame, text="ENTER EMPLOYEE NAME: ")
    nameLabel.pack()

    employeeName = tk.Entry(mainFrame)
    employeeName.pack()

    proceedFrame = tk.Frame(mainFrame, pady=20)
    proceedFrame.pack()

    proceedBtn = tk.Button(proceedFrame, text="PROCEED TO FACE CAPTURE", pady=10, padx=10, command=lambda: saveImage(employeeName.get(), root))
    proceedBtn.pack()

    root.mainloop()

def success(root, type, msg):
    global success_label
    text = ""
    if type == "time out":
        text = "TIME OUT: " + msg
    else:
        text = "TIME IN: " + msg

    success_label.config(text=text)  # Update the success label text

    # Close the success message after 3 seconds
    root.after(3000, lambda: success_label.config(text=""))



def detectFace(timeType):
    global success_label  # Declare success_label as global
    global root 
    delay_duration = 5  # seconds

    # Capture start time
    start_time = time.time()
    path = "face_images"
    images = []
    classNames = []
    mylist = os.listdir(path)
    for cl in mylist:
        curImgPath = f'{path}/{cl}'
        print("Loading image:", curImgPath)  # Debugging print
        try:
            curImg = cv2.imread(curImgPath)
            if curImg is not None:
                images.append(curImg)
                classNames.append(os.path.splitext(cl)[0])
            else:
                print("Error loading image:", curImgPath)
        except Exception as e:
            print("Error loading image:", curImgPath, e)

    if not images:
        print("No images found in the 'face_images' directory.")
        return

    encoded_face_train = findEncodings(images)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening webcam.")
        return

    cam = True
    while cam:
        success, img = cap.read()

        if not success:
            print("Error capturing webcam frame.")
            continue

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        faces_in_frame = face_recognition.face_locations(imgS)
        encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)

        if not encoded_faces:
            print("No face encodings found.")
            continue

        detected = False  # Flag to indicate whether a recognized face was detected

        for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
            matches = face_recognition.compare_faces(encoded_face_train, encode_face)
            faceDist = face_recognition.face_distance(encoded_face_train, encode_face)

            if len(faceDist) == 0:
                print("No face distances calculated.")
                continue

            matchIndex = np.argmin(faceDist)
          
            if matches[matchIndex]:
                name = classNames[matchIndex].upper().lower()
                detected = True
                y1, x2, y2, x1 = faceloc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                current_time = time.time()
                if current_time - start_time >= delay_duration:
                    cam = False
                    if not isValidImageFormat(name):
                        print("Image format is not valid:", name)
                    else:
                        if timeType == "time out":
                            if not checkIfAlreadyExists(name, 'TIME OUT'):
                                timeOut(name)
                            else:
                                print("Already timed out today:", name)
                        else:
                            if not checkIfAlreadyExists(name, 'TIME IN'):
                                timeIn(name)
                            else:
                                print("Already timed in today:", name)
                    break

        if not detected:  # No recognized face was detected
            cv2.putText(img, "Not registered as employee", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q') or not cam:
            closeWebcamWindow()  # Close the webcam window
            break

    cap.release()
    cv2.destroyAllWindows()

def isValidImageFormat(name):
    image_path = os.path.join('face_images', name + '.jpg')
    return os.path.exists(image_path)

def closeWebcamWindow():
    cv2.destroyAllWindows()



def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_faces = face_recognition.face_encodings(img)

        if encoded_faces:
            encodeList.append(encoded_faces[0])
    return encodeList

def saveImage(name, root):
    global success_label  # Declare success_label as global
    root.destroy()
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Open the camera
    cap = cv2.VideoCapture(0)

    # Introduce a delay of 2 seconds to ensure face quality
    delay_duration = 5  # seconds

    # Capture start time
    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        current_time = time.time()

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame with rectangles
        cv2.imshow('Face Detection', frame)

        # Capture a face image after the specified delay duration
        if current_time - start_time >= delay_duration and len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_image = frame[y:y+h, x:x+w]
            image_path = os.path.join('face_images', name + '.jpg')
            cv2.imwrite(image_path, face_image)
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()
    success(root, type="employee added", msg=f"Employee {name} added successfully.")


def timeIn(name):
    global success_label
    global root
    now = datetime.now()
    time_value = now.strftime('%I:%M:%S:%p')
    date = now.strftime('%d-%B-%Y')
    recordData = f'{name}, {time_value}, {date}, TIME IN'
    baseData = f'{name},{date}'

    csv_file_path = 'Attendance.xls'

    if not checkIfAlreadyExists(name, 'TIME IN'):
        if name.lower() == "carl" or name.lower() == "adrianvyne":
            # Subtract 15 minutes for specific users
            adjusted_time = now - timedelta(minutes=15)
            time_value = adjusted_time.strftime('%I:%M:%S:%p')

        with open(csv_file_path, 'a') as f:
            csv_line = f'{name}	{time_value}	{date}	TIME IN\n'
            f.write(csv_line)

        success_label.config(text=f"Time In Successful: {recordData}")  # Update success_label with success message
    else:
        success_label.config(text=f"Already Timed In Today: {baseData}") 

        face_image = cv2.resize(face_image, (300, 300))  # Resize the image
        image_path = os.path.join('face_images', name + '.jpg')
        cv2.imwrite(image_path, face_image)  # Save the resized image
        success(root, type="employee added", msg=f"Employee {name} added successfully.")
        
        # Call the timeIn function to record the entry time
        recordTime(name, "TIME IN")

def timeOut(name):
    global success_label
    global root
    now = datetime.now()
    time_value = now.strftime('%I:%M:%S:%p')
    date = now.strftime('%d-%B-%Y')
    recordData = f'{name}, {time_value}, {date}, TIME OUT'
    baseData = f'{name},{date}'

    csv_file_path = 'Attendance.xls'

    if not checkIfAlreadyExists(name, 'TIME OUT'):
        with open(csv_file_path, 'a') as f:
            csv_line = f'{name}	{time_value}	{date}	TIME OUT\n'
            f.write(csv_line)
        
        # Update the success label widget on the UI
        success_label.config(text=f"Time Out Successful: {recordData}")
    else:
        # Update the success label widget on the UI
        success_label.config(text=f"Already Timed Out Today: {baseData}")


        face_image = cv2.resize(face_image, (300, 300))  # Resize the image
        image_path = os.path.join('face_images', name + '.jpg')
        cv2.imwrite(image_path, face_image)  # Save the resized image
        success_label.config(text=f"Time Out Successful: {recordData}")
        
        # Call the timeOut function to record the exit time
        recordTime(name, "TIME OUT")



def recordTime(name, timeType):
    now = datetime.now()
    time_value = now.strftime('%I:%M:%S:%p')
    date = now.strftime('%d-%B-%Y')
    recordData = f'{name}, {time_value}, {date}, {timeType}'
    csv_file_path = 'Attendance.xls'

    with open(csv_file_path, 'a') as f:
        csv_line = f'{name}	{time_value}	{date}	{timeType}\n'
        f.write(csv_line)

    success_label.config(text=f"{timeType.capitalize()} Successful: {recordData}")


def checkIfAlreadyExists(name, timeType):
    now = datetime.now()
    date = now.strftime('%d-%B-%Y')
    baseData = f'{name}	{date}	{timeType}'
    
    csv_file_path = 'Attendance.xls'

    with open(csv_file_path, 'r') as f:
        myDataList = f.readlines()
        for line in myDataList:
            entry = line.strip().split(',')
            if len(entry) >= 4 and entry[0] == name and entry[2] == date and entry[3] == timeType:
                return True
    return False


if __name__ == "__main__":
    main()