import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

X = np.load('image.npz')['arr_0']
Y = pd.read_csv("labels.csv")["labels"]

classes = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J', "K",
"L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
"Y", "Z"]

nclasses = len(classes)

X_train, X_test, Y_train, Y_test = tts(X,Y,train_size= 7500, test_size= 2500, random_state= 0)

X_trainScaled = X_train/255.0
X_testScaled = X_test/255.0

lr = LogisticRegression(solver= 'saga', multi_class= 'multinomial').fit(X_trainScaled, Y_train)

def GetPrediction(Image):
    imagePIL = Image.open(Image)
    ImageBw = imagePIL.convert('L')
    ImageBwResized = ImageBw.resize(28,28,Image.ANTIALIAS)
    PixelFilter = 20
    MinPixel = np.percentile(ImageBwResized, PixelFilter)
    ImageBwResizedInvertedScaled = np.clip(ImageBwResized - MinPixel, 0, 255)
    MaxPixel = np.max(ImageBwResized)
    ImageBwResizedInvertedScaled = np.asarray(ImageBwResizedInvertedScaled/MaxPixel)
    TestSample = np.array(ImageBwResizedInvertedScaled).reshape(1, 784)
    TestPredict = lr.predict(TestSample)
    return TestPredict[0]