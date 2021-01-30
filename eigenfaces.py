from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import glob
import re

#Read Images
#Make each img as one vector
dataset_path  = r'D:\suneel\2\ML\A2\yalefaces'
image_list = os.listdir(dataset_path)

sample = plt.imread(r'D:\suneel\2\ML\A2\yalefaces\subject01.normal.jpg')
height = sample.shape[0]
width = sample.shape[1]
total_1d_imgs = np.ndarray(shape=(len(image_list), height*width), dtype=np.float64)

count = 0
for image in glob.glob(r"D:\suneel\2\ML\A2\yalefaces\*"):
    img_3d = plt.imread(image)
    total_1d_imgs[count, :] = np.array(img_3d, dtype='float64').flatten()
    count= count+1

#Calculate mean of all images
### Mean required to find the common features in the images

print("Mean Image:")
mean_image_vector = np.zeros((1,height*width), dtype=np.float64)
mean_image_vector = np.average(total_1d_imgs, axis=0)
img = mean_image_vector.reshape(height,width)
plt.imshow(img, cmap='jet')

#Subtract Mean image from all the images and store it in matrix called mean_subtracted_images(Normalized images)


mean_subtracted_images = np.ndarray(shape=(len(image_list), height*width))

for i in range(len(image_list)):
    mean_subtracted_images[i] = np.subtract(total_1d_imgs[i], mean_image_vector)

#Compute Covarience matrix
cov_matrix = np.cov(mean_subtracted_images)
cov_matrix = np.divide(cov_matrix,float(len(image_list)))

#Calculate Eigen values and Eigen vectores of Covarience matrix

eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

eigen_val_vect_pair = []
for index in range(len(eigen_values)):
    eigen_val_vect_pair.append((eigen_values[index], eigen_vectors[:,index]))
    
for i in range(len(eigen_values)):
    for j in range(len(eigen_values)):
        if(eigen_val_vect_pair[i][0]>eigen_val_vect_pair[j][0]):
            temp = eigen_val_vect_pair[i]
            eigen_val_vect_pair[i] = eigen_val_vect_pair[j]
            eigen_val_vect_pair[j] = temp

# print(eigen_val_vect_pair)
sorted_eig_val  = [eigen_val_vect_pair[index][0] for index in range(len(eigen_values))]
sorted_eig_vect = [eigen_val_vect_pair[index][1] for index in range(len(eigen_values))]
# print(len(sorted_eig_val))

var_comp_sum = np.cumsum(sorted_eig_val)/sum(sorted_eig_val)


num_comp = range(1,len(sorted_eig_val)+1)
plt.xlabel('Principal Components')
plt.ylabel('Variance')
plt.scatter(num_comp, var_comp_sum)
plt.show()

#Take eigen vectors which are having highest varience
required_eig_vectors = np.array(sorted_eig_vect[3:14]).transpose()

proj_data = np.dot(total_1d_imgs.transpose(),required_eig_vectors)
proj_data = proj_data.transpose()

for i in range(proj_data.shape[0]):
    img = proj_data[i].reshape(height,width)
    plt.subplot(4,4,1+i)
    plt.imshow(img, cmap='jet')
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
plt.show()

#Finding weights for each traning image
weight = np.array([np.dot(proj_data,i) for i in mean_subtracted_images])

##Recognition Part
total_test = 0
matched = 0
test_img_path = r'D:\suneel\2\ML\A2\test\\'
for image_test in os.listdir(test_img_path):
    img_with_path = test_img_path+image_test
    test_img = plt.imread(img_with_path)
    test_1d_img = np.array(test_img, dtype='float64').flatten()

    mean_subtracted_test_img = np.subtract(test_1d_img, mean_image_vector)

    test_weight = np.dot(proj_data, mean_subtracted_test_img)

    weight_diff = weight-test_weight

    norms = np.linalg.norm(weight_diff, axis=1)
    index = np.argmin(norms)
    
    #This is converting subject01.normal to '01'
    temp = re.compile("([a-zA-Z]+)([0-9]+)") 
    res = temp.match(image_test.split('.')[0]).groups() 

    total_test = total_test+1
    if int(res[1])==index+1:
        matched = matched +1
        
print("In Total {} test Images {} images are matched and Final Matching Accuracy = {} ".format(total_test,matched,matched/total_test*100))