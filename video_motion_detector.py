import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import model_from_json

class FacialExpressionModel(object):

    EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]


# The face detector network is loaded using cv2.dnn.readNetFromCaffe and the model's layers and weights as passed its arguments.
modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Load the age detector model from disk
model_age = load_model('models/age_model.h5')

model_emotion = FacialExpressionModel("models/emotion_model.json", "models/emotion_model.h5")

# Age ranges are defined
ranges = ['1-2','3-9','10-20','21-27','28-45','46-65','66-116']

# Capturing video or webcam
cap = cv2.VideoCapture('myVideo.mp4')

# Final result is going to be written in the disk
result = cv2.VideoWriter('output/model.mov', 
                        cv2.VideoWriter_fourcc(*'MJPG'),
                        30, (1620,1080))

while True:
    
    # Read the frame
    _, img = cap.read()

    # Height and width of the image are extracted
    h, w = img.shape[:2]

    # To achieve the best accuracy I ran the model on BGR images resized to 300x300 
    # applying mean subtraction of values (104, 177, 123) for each blue, green and red channels correspondingly.
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,(300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)

    # Face detection is performed
    faces = net.forward()

    # For each face detected...
    for j in range(faces.shape[2]):
        confidence = faces[0, 0, j, 2]
        # If the confidence is above a certain threshold
        if confidence > 0.5:
            box = faces[0, 0, j, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            # Face is extracted
            img_face = img[y:y1, x:x1]
            # Tranformed to grayscale
            face = cv2.cvtColor(img_face,cv2.COLOR_BGR2GRAY)
            # Resized and reshaped to fit the input layer of the network
            face_age = cv2.resize(face,(200,200))
            face_age = face_age.reshape(1,200,200,1)
            # Normalized
            normalizer = ImageDataGenerator(rescale=1./255)
            facenorm = normalizer.flow(face_age)

            face_emotion = cv2.resize(face, (48, 48))
            emotion = model_emotion.predict_emotion(face_emotion[np.newaxis, :, :, np.newaxis])

            # Prediction is performed
            age = model_age.predict(facenorm)
            # A bounding box is drawn surrounding the face
            cv2.rectangle(img, (x, y), (x1, y1), (0,0,255), 2)
            # The estimated age range is printed
            #cv2.putText(img,ranges[np.argmax(age)],(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.8,(255,255,255),2)
            cv2.putText(img, emotion, (x1, y1+30), cv2.FONT_HERSHEY_DUPLEX,0.8, (255, 255, 0), 2)

            
    # Display
    cv2.imshow('Person Detector', img)
    result.write(img)

    # Stop if escape key is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
        
# Release the VideoCapture object
cap.release()
result.release()
cv2.destroyAllWindows()