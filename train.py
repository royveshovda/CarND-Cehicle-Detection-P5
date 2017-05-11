import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC, SVC
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

#train test split.
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2)

parameter_search = False

if parameter_search:
    parameters = [
      #{'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], 'kernel': ['linear']}#,
      {'C': [0.000001, 0.00001, 0.0001, 0.001], 'kernel': ['linear']}#,
    #  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
     ]
    #parameters = {'kernel':('linear', 'rbf'), 'C':[1]}
    svr = SVC()
    clf = GridSearchCV(svr, parameters, n_jobs=-1, verbose=2)
    clf.fit(X_train, y_train)
    print(clf.best_params_)
else:
    # Works, but slow
    svc = SVC(kernel='rbf', C=10, gamma=0.0001, verbose=2)

    #svc = LinearSVC(C=0.001)

    # Train the SVC
    svc.fit(X_train, y_train)

    #save the model
    filename_model = 'model.dat'
    pickle.dump(svc, open(filename_model, 'wb'))

    #scale the scaling factor
    filename_scaler = 'scaler.dat'
    pickle.dump(X_scaler, open(filename_scaler, 'wb'))

    #prints the accuracy
    print('Test Accuracy of SVC = ', svc.score(X_test, y_test))
    print('Training OK!')
