*****************************************************************************************************************************************************
							GUIDE TO RUN THE PROJECT
*****************************************************************************************************************************************************

1. Open the "Code" folder and move the "Main.ipynb" to your Jupyter notebook. Place this jupyter source file under a new folder. (You can keep any name to this newly created folder)

2. Open the "Data" folder and place all the images (icons + test images) in a new folder named "data" directly under C drive in local machine. (C:\data)

3. Run the "Main.ipynb" file in Jupyter notebook. The libraries need to be installed are: Tensorflow, Keras, NumPy and Matplolib. Following are the commands:

		from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
		from keras.preprocessing.image import ImageDataGenerator,image 
		from keras.layers import Dense,Activation,Flatten,Dropout
		from keras.models import Sequential,Model,load_model
		from tensorflow.keras import optimizers
		from keras.callbacks import ModelCheckpoint,EarlyStopping
		import numpy as np
		import matplotlib.pyplot as plt

4. After successful running of the "Main.ipynb".

5. An image of the rice grain is passed as input to the desktop application which returns the result with an accuracy upto 99%.

