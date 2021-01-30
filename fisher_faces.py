from matplotlib import pyplot as plt
import numpy as np
import os
import glob
import re
from scipy.linalg import eigh
number_of_classes = 15
number_of_samples_in_class = 4

#Read Images
#Make each img as one vector
dataset_path  = r'/yalefaces_train/'
image_list = os.listdir(dataset_path)

# Using one sample image to record the dimensions
sample = plt.imread(r'/yalefaces_train/subject01.normal.jpg')
height = sample.shape[0]
width = sample.shape[1]
total_1d_imgs = np.ndarray(shape=(len(image_list), height*width), dtype=np.float64)
#Creating a variable to store all the images as 1*n dimensions 
count = 0
for image in glob.glob(r"/yalefaces_train/*"):
    img_3d = plt.imread(image)
    total_1d_imgs[count, :] = np.array(img_3d, dtype='float64').flatten()
    count= count+1


#Calculate mean of all images
### Mean required to find the common features in the images

print("Mean images")
all_images_mean_vector = np.zeros((1,height*width), dtype=np.float64)
all_images_mean_vector = np.average(total_1d_imgs, axis=0)
img = all_images_mean_vector.reshape(height,width)
plt.imshow(img, cmap='jet')


#Read class wise images    
one_class_imgs = np.ndarray(shape=(number_of_samples_in_class, height*width), dtype = np.float64)
class_wise_imgs = np.ndarray(shape=(number_of_classes, number_of_samples_in_class, height*width), dtype = np.float64)
count = 0
for i in range(number_of_classes):
    count = 0
    for image_train in os.listdir(dataset_path):
        img_with_path = dataset_path+image_train
        #print(img_with_path)
        temp = re.compile("([a-zA-Z]+)([0-9]+)") 
        res = temp.match(image_train.split('.')[0]).groups() 

        if int(res[1]) == i+1:
            t_img = plt.imread(img_with_path)
            #print(np.array(t_img, dtype = 'float64').flatten())
            one_class_imgs[count,:] = np.array(t_img, dtype = 'float64').flatten()
            count = count +1
        
    class_wise_imgs[i,:,:] = one_class_imgs[:]


#Calculate mean image for each class
class_wise_mean_imgs = np.ndarray(shape=(number_of_classes, height*width), dtype = np.float64)
for i in range(len(class_wise_imgs)):
    class_wise_mean_imgs[i,:] = np.average(class_wise_imgs[i], axis=0)
    
for i in range(number_of_classes):
    img = class_wise_mean_imgs[i].reshape(height,width)
    plt.subplot(4,4,1+i)
    plt.imshow(img, cmap='jet')
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
plt.show()

# Calculate between-class scatter matrix
print("Class wise mean subtracted image")
class_wise_mean_subtracted_imgs = np.ndarray(shape=(number_of_classes, height*width), dtype = np.float64)
for i in range(number_of_classes):
    class_wise_mean_subtracted_imgs[i,:] = np.subtract(class_wise_mean_imgs[i], all_images_mean_vector)
    
for i in range(number_of_classes):
    img = class_wise_mean_subtracted_imgs[i].reshape(height,width)
    plt.subplot(4,4,1+i)
    plt.imshow(img, cmap='jet')
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
plt.show()
scatter_between_class= np.zeros((15,15), dtype=np.float64)
#scatter_between_class= np.zeros((height*width,height*width), dtype=np.float64)  # Uncomment this one when using HPC
for i in range(number_of_classes):
    scatter_between_class = scatter_between_class+ np.dot(class_wise_mean_subtracted_imgs[i,:],class_wise_mean_subtracted_imgs[i,:].transpose())*number_of_samples_in_class
#print(scatter_between_class)

# Calculate scatter within class matrix
scatter_within_class = np.zeros((15,15), dtype=np.float64)
#scatter_within_class = np.zeros((height*width,height*width), dtype=np.float64)   # Uncomment this one when using HPC
for i in range(number_of_classes):
    for j in range(number_of_samples_in_class):
        Tr=np.subtract(class_wise_imgs[i,j,:],class_wise_mean_imgs[i,:])
        scatter_within_class = scatter_within_class + np.dot(Tr,Tr.transpose())
#print(scatter_within_class)

        ######################################
############ Forming W matrix using PCA   ################
        ######################################
cov_matrix = np.cov(class_wise_mean_subtracted_imgs)
cov_matrix = np.divide(cov_matrix,float(len(image_list)/number_of_samples_in_class))

#Calculate Eigen values and Eigen vectores of Covariance matrix

eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
#Storing them in pair
eigen_val_vect_pair = []
for index in range(len(eigen_values)):
    eigen_val_vect_pair.append((eigen_values[index], eigen_vectors[:,index]))
    
for i in range(len(eigen_values)):
    for j in range(len(eigen_values)):
        if(eigen_val_vect_pair[i][0]>eigen_val_vect_pair[j][0]):
            temp = eigen_val_vect_pair[i]
            eigen_val_vect_pair[i] = eigen_val_vect_pair[j]
            eigen_val_vect_pair[j] = temp

# Sorting eigen values and eigne vectors
sorted_eig_val  = [eigen_val_vect_pair[index][0] for index in range(len(eigen_values))]
sorted_eig_vect = [eigen_val_vect_pair[index][1] for index in range(len(eigen_values))]


#Take eigen vectors which are having highest variance
required_eig_vectors_pca = np.array(sorted_eig_vect[:15]).transpose()


############################################################################ Method 1 ###############################################################

        ######################################
############ Forming W matrix using LDA   ################
        ######################################


##### Original method as in paper #######
#A=np.dot(np.dot(scatter_between_class,required_eig_vectors_pca.transpose()),required_eig_vectors_pca)
#B=np.dot(np.dot(scatter_within_class,required_eig_vectors_pca.transpose()),required_eig_vectors_pca)
#print(A.shape)
#print(B.shape)
#eigen_values, eigen_vectors = eigh(A,B,eigvals_only=False)

#### Using pseudo inverse approach #######

eigen_values, eigen_vectors = np.linalg.eig(np.linalg.pinv(scatter_within_class)*scatter_between_class)
# Forming pairs of eigen values and eignen vectors
eigen_val_vect_pair = []
for index in range(len(eigen_values)):
    eigen_val_vect_pair.append((eigen_values[index], eigen_vectors[:,index]))
    
for i in range(len(eigen_values)):
    for j in range(len(eigen_values)):
        if(eigen_val_vect_pair[i][0]>eigen_val_vect_pair[j][0]):
            temp = eigen_val_vect_pair[i]
            eigen_val_vect_pair[i] = eigen_val_vect_pair[j]
            eigen_val_vect_pair[j] = temp

# Sorting
sorted_eig_val  = [eigen_val_vect_pair[index][0] for index in range(len(eigen_values))]
sorted_eig_vect = [eigen_val_vect_pair[index][1] for index in range(len(eigen_values))]
required_eig_vectors = np.array(sorted_eig_vect[:15]).transpose()

#### FINAL W MATRIX #####
w_opt=np.dot(required_eig_vectors_pca,required_eig_vectors)

### Forming feature space ###
proj_data = np.dot(class_wise_mean_imgs.transpose(),w_opt)
proj_data = proj_data.transpose()
weight = np.array([np.dot(proj_data,i) for i in class_wise_mean_subtracted_imgs])

################## Testing ##################

total_test = 0
matched = 0
test_img_path = r'/test_fisher/'
for image_test in os.listdir(test_img_path):
    img_with_path = test_img_path+image_test
    test_img = plt.imread(img_with_path)
    test_1d_img = np.array(test_img, dtype='float64').flatten()
    mean_subtracted_test_img = np.subtract(test_1d_img, all_images_mean_vector)
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
print("In Total {} test Images {} images are matched and Final Matching Accuracy (Method 1) = {} ".format(total_test,matched,matched/total_test*100))

############################################################################ Method 2 ###############################################################
print("Using second proposed method")

### Forming W using scatter between class matrix as discussed in the paper ###
eigen_values, eigen_vectors = np.linalg.eig(scatter_between_class)
eigen_val_vect_pair = []
for index in range(len(eigen_values)):
    eigen_val_vect_pair.append((eigen_values[index], eigen_vectors[:,index]))
    
for i in range(len(eigen_values)):
    for j in range(len(eigen_values)):
        if(eigen_val_vect_pair[i][0]>eigen_val_vect_pair[j][0]):
            temp = eigen_val_vect_pair[i]
            eigen_val_vect_pair[i] = eigen_val_vect_pair[j]
            eigen_val_vect_pair[j] = temp

sorted_eig_val  = [eigen_val_vect_pair[index][0] for index in range(len(eigen_values))]
sorted_eig_vect = [eigen_val_vect_pair[index][1] for index in range(len(eigen_values))]
required_eig_vectors_between_class = np.array(sorted_eig_vect[:45]).transpose()
proj_data = np.dot(class_wise_mean_imgs.transpose(),required_eig_vectors_between_class)
proj_data = proj_data.transpose()
weight = np.array([np.dot(proj_data,i) for i in class_wise_mean_subtracted_imgs])
#### Testing ####
total_test = 0
matched = 0
test_img_path = r'/test_fisher/'
for image_test in os.listdir(test_img_path):
    img_with_path = test_img_path+image_test
    test_img = plt.imread(img_with_path)
    test_1d_img = np.array(test_img, dtype='float64').flatten()
    mean_subtracted_test_img = np.subtract(test_1d_img, all_images_mean_vector)
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
        
print("In Total {} test Images {} images are matched and Final Matching Accuracy (Method 2) = {} ".format(total_test,matched,matched/total_test*100))



