import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pickle

car = np.load("car_features.data")
non_car = np.load("non_car_features.data")

#comibne both car and non car image
all_data_x = np.vstack([car,non_car]).astype(np.float64)
all_data_y = np.hstack((np.ones(len(car)),
              np.zeros(len(non_car))))

#feature scaling
X_scaler = StandardScaler().fit(all_data_x)
# Apply the scaler to X
scaled_X = X_scaler.transform(all_data_x)

#assign to y
y = all_data_y;

rand_state = 24

#train test split. Also randomises.
#X_rem = train + validation. I dont use validation set.
X_rem, X_test, y_rem, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

#The X_rem is again split into train test
X_train, X_valid, y_train, y_valid = train_test_split(
    X_rem, y_rem, test_size=0.2, random_state=rand_state)

# Use a linear SVC (support vector classifier)
svc = LinearSVC(random_state=rand_state,verbose=False,max_iter=2000)
# Train the SVC
svc.fit(X_rem, y_rem)

#save the model
filename_model = 'model.dat'
pickle.dump(svc, open(filename_model, 'wb'))

#scale the scaling factor
filename_scaler = 'scaler.dat'
pickle.dump(X_scaler, open(filename_scaler, 'wb'))

#prints the accuracy
print('Test Accuracy of SVC = ', svc.score(X_test, y_test))

print('Training OK!')
