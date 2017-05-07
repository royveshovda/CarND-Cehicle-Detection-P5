import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

path = 'vehicles'
car_files = [os.path.join(dirpath, f)
    for dirpath, dirnames, files in os.walk(path)
    for f in files if f.endswith('.png')]

path2 = 'non-vehicles'
non_car_files = [os.path.join(dirpath, f)
    for dirpath, dirnames, files in os.walk(path2)
    for f in files if f.endswith('.png')]

#print number of files availabel
print(len(car_files))
print(len(non_car_files))


#Resize to 64,64
def data_read_images_from_files(file_names):
    images = [];
    for file in file_names:
        image = cv2.cvtColor(cv2.imread(file),cv2.COLOR_BGR2RGB);
        image = cv2.resize(image,dsize=(64,64))
        images.append(image);
    return np.array(images)


# Define a function to return few characteristics of the dataset
def data_look(car_list, notcar_list):
    data_dict = {}
    data_dict["n_cars"] = len(car_list)
    data_dict["n_notcars"] = len(notcar_list)
    example_img = car_list[0]
    data_dict["image_shape"] = example_img.shape
    data_dict["data_type"] = example_img.dtype
    return data_dict

#all car images
car = data_read_images_from_files(car_files);
#all non car images
non_car = data_read_images_from_files(non_car_files)
#get infor about the data
data_info = data_look(car, non_car)

#print the data info
print('Your function returned a count of',
      data_info["n_cars"], ' cars and',
      data_info["n_notcars"], ' non-cars')
print('of size: ', data_info["image_shape"], ' and data type:',
      data_info["data_type"])

#dunp data back to the file system as numpy array
car.dump("car_images.dat")
non_car.dump("non_car_images.dat")

print('Loaded OK!')
