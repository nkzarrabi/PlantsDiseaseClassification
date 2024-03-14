from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage

import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import tensorflow as tf 

# Assuming TensorFlow 2.x, eager execution is enabled by default.
model = load_model("classify/first_one.h5")

IMG_WIDTH = 224
IMG_HEIGHT = 224

labels = {
    0: "Corn Gray leaf spot", 1: "Common rust Corn Maize", 
    2: "Northern Leaf Blight Corn Maize", 3: "Healthy Maize Corn", 
    4: "Early blight Potato leaf", 5: "Late blight Potato Leaf", 
    6: "healthy Potato Leaf", 7: "Bacterial spot of Tomato", 
    8: "Leaf Mold of Tomato", 9: "Yellow Leaf Curl Virus Tomato", 
    10: "Mosaic virus leaf Tomato", 11: "Healthy Tomato Leaf"
}

def index(request):
    return render(request, "index.html")

def predictImage(request):
    print(request.POST.dict())
    fileObj = request.FILES["document"]
    fs = FileSystemStorage()
    filePathName = fs.save(fileObj.name, fileObj)
    filePathName = fs.url(filePathName)

    test_image = "." + filePathName

    img = image.load_img(test_image, target_size=(IMG_WIDTH, IMG_HEIGHT, 3))
    img = image.img_to_array(img)
    img = img / 255
    x = np.expand_dims(img, axis=0)
    
    proba = model.predict(x)
    top_3 = np.argsort(proba[0])[:-4:-1]
    max_proba = np.max(proba)
    if max_proba > 0.5:
        for i in range(1):
            label_pred = labels[top_3[i]]
            acc_pred = proba[0][top_3[i]] * 100
    else:
        message = "This image is not of leaf please try other one."

    context = {
        "filePathName": filePathName,
        "prediction": label_pred,
        "Accuracy": acc_pred,
    }
    return render(request, "test.html", context)
