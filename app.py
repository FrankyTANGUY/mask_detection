# Import des bibliothèques
import cv2
import numpy as np
import keras

#Import du modèle IA
model = keras.models.load_model('./Mask.h5')

#Application de détection de masque en temps réel
cap = cv2.VideoCapture(0)
while True:
    # Reading the frame from the camera
    _,frame = cap.read()
    #Flipping the frame to see same side of yours
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    image_test=cv2.resize(frame, (224,224))

    image_test= np.expand_dims(image_test,axis=0)
    image_test.shape
    predictions = model.predict(image_test)
    label_predit=np.argmax(predictions)
    if label_predit==0 :
        message = "No Mask"
        cv2.putText(frame, message, (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255))
    else :
        message = "Masque"
        cv2.putText(frame, message, (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0))
    cv2.imshow("Mask Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# Release the camera and all resources
cap.release()
cv2.destroyAllWindows()