# USAGE
# python liveness_demo.py --model liveness.model --le le.pickle --detector face_detector

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from imutils.video import FPS
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import face_recognition
import argparse
import imutils
import pickle
import dlib
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m_t", "--model", type=str, default="output/liveness_rose.model",
                help="path to trained model")
ap.add_argument("-l_l", "--le_liveness", type=str, default="output/le_liveness_rose.pickle",
                help="path to label encoder")
ap.add_argument("-d", "--detector", type=str, default="models/face_detector",
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-o", "--output", type=str, default="output/liveness_face_recognition.avi",
                help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
                help="whether or not to display output frame to screen")
ap.add_argument("-l_i", "--le_id", type=str, default="output/le_id.pickle",
                help="path to label encoder")
ap.add_argument("-p", "--shape-predictor", default="models/face_landmark/shape_predictor_68_face_landmarks.dat",
                help="path to facial landmark predictor")
ap.add_argument("-e", "--embeddings", default="output/encodings.pickle",
                help="path to serialized db of facial embeddings")
args = vars(ap.parse_args())


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load the liveness detector model and label encoder from disk
print("[INFO] loading liveness detector...")
model = load_model(args["model"])
le_liveness = pickle.loads(open(args["le_liveness"], "rb").read())

# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())

# create the facial landmark predictor and the face aligner
predictor = dlib.shape_predictor(args["shape_predictor"])

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.22
EYE_AR_CONSEC_FRAMES = 3

TOLERANCE = 0.45
FACE_THRESH = 7

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# initialize the frame counters and the total number of blinks
names = list(set(data["names"]))
count = [0] * len(names)
COUNTER = dict(zip(names, count))
TOTAL = COUNTER.copy()

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
writer = None
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()

# initialize the recognition mode(0:liveness detection on, 1:liveness detection off)
rec_mod = 0
status_l = "OFF"
status_b = "OFF"
sleep_flag = 0

# define colors
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 600 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the face and extract the face ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the detected bounding box does fall outside the
            # dimensions of the frame
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            # extract the face ROI
            face = frame[startY:endY, startX:endX]

            if rec_mod == 0:
                # encode a new face image to 128-D embedding
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb, [(startY, endX, endY, startX)])

                # attempt to match the new face in the input image to our known
                # encodings
                matches = face_recognition.compare_faces(data["encodings"], encodings[0], tolerance=TOLERANCE)
                name = "Unknown"

                # check to see if we have found a match
                if True in matches:
                    # find the indexes of all matched faces then initialize a
                    # dictionary to count the total number of times each face
                    # was matched
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}

                    # loop over the matched indexes and maintain a count for
                    # each recognized face face
                    for i in matchedIdxs:
                        name = data["names"][i]
                        counts[name] = counts.get(name, 0) + 1

                    # determine the recognized face with the largest number
                    # of votes (note: in the event of an unlikely tie Python
                    # will select first entry in the dictionary)
                    name = max(counts, key=counts.get)
                    if counts[name] < FACE_THRESH:
                        name = "Unknown"

                # draw the label and bounding box on the frame
                label = "{}".format(name)
                y = startY - 10 if startY - 10 > 10 else startY + 20
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), GREEN, 1)

            elif rec_mod == 1:
                # extract the face ROI and then pre-process it in the exact
                # same manner as our training data
                face = cv2.resize(face, (64, 64))
                face = face.astype("float") / 255.0
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)

                # pass the face ROI through the trained liveness detector
                # model to determine if the face is "real" or "fake"
                preds = model.predict(face)[0]
                j = np.argmax(preds)
                label = le_liveness.classes_[j]

                # perform classification to recognize the face
                if label == "Real":
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(rgb, [(startY, endX, endY, startX)])

                    # attempt to match each face in the input image to our known
                    # encodings
                    matches = face_recognition.compare_faces(data["encodings"], encodings[0], tolerance=TOLERANCE)
                    name = "Unknown"

                    # check to see if we have found a match
                    if True in matches:
                        # find the indexes of all matched faces then initialize a
                        # dictionary to count the total number of times each face
                        # was matched
                        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                        counts = {}

                        # loop over the matched indexes and maintain a count for
                        # each recognized face face
                        for i in matchedIdxs:
                            name = data["names"][i]
                            counts[name] = counts.get(name, 0) + 1

                        # determine the recognized face with the largest number
                        # of votes (note: in the event of an unlikely tie Python
                        # will select first entry in the dictionary)
                        name = max(counts, key=counts.get)
                        if counts[name] < FACE_THRESH:
                            name = "Unknown"

                    # draw the label and bounding box on the frame
                    label = "{}: {:.4f} {}".format(label, preds[j], name)
                    y = startY - 10 if startY - 10 > 10 else startY + 20
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), GREEN, 1)

                else:
                    # draw the label and bounding box on the frame
                    label = "{}: {:.4f}".format(label, preds[j])
                    y = startY - 10 if startY - 10 > 10 else startY + 20
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), RED, 1)

            elif rec_mod == 2:
                # extract the face ROI and then pre-process it in the exact
                # same manner as our training data
                face = cv2.resize(face, (64, 64))
                face = face.astype("float") / 255.0
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)

                # pass the face ROI through the trained liveness detector
                # model to determine if the face is "real" or "fake"
                preds = model.predict(face)[0]
                j = np.argmax(preds)
                label = le_liveness.classes_[j]

                # perform classification to recognize the face
                if label == "Real":
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(rgb, [(startY, endX, endY, startX)])

                    # attempt to match each face in the input image to our known
                    # encodings
                    matches = face_recognition.compare_faces(data["encodings"], encodings[0], tolerance=TOLERANCE)
                    name = "Unknown"

                    # check to see if we have found a match
                    if True in matches:
                        # find the indexes of all matched faces then initialize a
                        # dictionary to count the total number of times each face
                        # was matched
                        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                        counts = {}

                        # loop over the matched indexes and maintain a count for
                        # each recognized face face
                        for i in matchedIdxs:
                            name = data["names"][i]
                            counts[name] = counts.get(name, 0) + 1

                        # determine the recognized face with the largest number
                        # of votes (note: in the event of an unlikely tie Python
                        # will select first entry in the dictionary)
                        name = max(counts, key=counts.get)
                        if counts[name] < FACE_THRESH:
                            name = "Unknown"

                    # if detected a person in database, execute eye-blink check
                    if name != "Unknown":
                        rect = dlib.rectangle(startX, startY, endX, endY)
                        shape = predictor(gray, rect)
                        shape = face_utils.shape_to_np(shape)

                        # extract the left and right eye coordinates, then use the
                        # coordinates to compute the eye aspect ratio for both eyes
                        leftEye = shape[lStart:lEnd]
                        rightEye = shape[rStart:rEnd]
                        leftEAR = eye_aspect_ratio(leftEye)
                        rightEAR = eye_aspect_ratio(rightEye)

                        # average the eye aspect ratio together for both eyes
                        ear = (leftEAR + rightEAR) / 2.0

                        # compute the convex hull for the left and right eye, then
                        # visualize each of the eyes
                        leftEyeHull = cv2.convexHull(leftEye)
                        rightEyeHull = cv2.convexHull(rightEye)
                        cv2.drawContours(frame, [leftEyeHull], -1, GREEN, 1)
                        cv2.drawContours(frame, [rightEyeHull], -1, GREEN, 1)

                        # check to see if the eye aspect ratio is below the blink
                        # threshold, and if so, increment the blink frame counter
                        if ear < EYE_AR_THRESH:
                            COUNTER[name] += 1

                            # draw the bounding box of the face along with the
                            # associated probability
                            text = "{}: {:.4f}".format(label, preds[j])
                            y = startY - 10 if startY - 10 > 10 else startY + 20
                            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
                            cv2.rectangle(frame, (startX, startY), (endX, endY), GREEN, 1)

                        # otherwise, the eye aspect ratio is not below the blink
                        # threshold
                        else:
                            # if the eyes were closed for a sufficient number of
                            # then increment the total number of blinks
                            if COUNTER[name] >= EYE_AR_CONSEC_FRAMES:
                                TOTAL[name] += 1

                                # draw the bounding box of the face along with the
                                # associated probability
                                text = "{}: {:.4f} {}: {}".format(label, preds[j], name, "Blinked!")
                                y = startY - 10 if startY - 10 > 10 else startY + 20
                                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                            GREEN, 2)
                                cv2.rectangle(frame, (startX, startY), (endX, endY), GREEN, 1)
                                cv2.putText(frame, "Blinks: {}".format(TOTAL[name]), (25, 400),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, RED, 2)
                                cv2.putText(frame, "Detected person: {}".format(name), (25, 430),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, RED, 2)
                                sleep_flag = 1
                            else:
                                # draw the bounding box of the face along with the
                                # associated probability
                                text = "{}: {:.4f}".format(label, preds[j])
                                y = startY - 10 if startY - 10 > 10 else startY + 20
                                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
                                cv2.rectangle(frame, (startX, startY), (endX, endY), GREEN, 1)

                            # reset the eye frame counter
                            COUNTER[name] = 0
                else:
                    # draw the label and bounding box on the frame
                    text = "{}: {:.4f}".format(label, preds[j])
                    y = startY - 10 if startY - 10 > 10 else startY + 20
                    cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), RED, 1)

    cv2.putText(frame, "Face recognition running", (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, BLUE, 2)
    cv2.putText(frame, "Liveness detection: {}".format(status_l), (25, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, BLUE, 2)
    cv2.putText(frame, "Blink detection: {}".format(status_b), (25, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, BLUE, 2)

    # # if the video writer is None *AND* we are supposed to write
    # # the output video to disk initialize the writer
    # if writer is None and args["output"] is not None:
    #     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    #     writer = cv2.VideoWriter(args["output"], fourcc, 20,
    #                              (frame.shape[1], frame.shape[0]), True)
    #
    # # if the writer is not None, write the frame with recognized
    # # faces to disk
    # if writer is not None:
    #     writer.write(frame)

    fps.update()
    # check to see if we are supposed to display the output frame to
    # the screen
    if args["display"] > 0:
        cv2.imshow("Face Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    if sleep_flag == 1:
        time.sleep(3)
        sleep_flag = 0

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # if the 't' key was pressed, change detection mode(turn on/off liveness detection)
    elif key == ord("t"):
        if rec_mod == 0:
            rec_mod = 1
            status_l = "ON"
        elif rec_mod == 1:
            rec_mod = 2
            status_b = "ON"
        elif rec_mod == 2:
            rec_mod = 0
            status_l = "OFF"
            status_b = "OFF"

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

# check to see if the video writer point needs to be released
if writer is not None:
    writer.release()
