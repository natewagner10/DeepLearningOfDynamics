#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# images from decoder 
PATH = os.getcwd()
data_path = PATH + '/deer_img'
data_path_folder = os.listdir(data_path)


# In[ ]:


# load in images
X_data = []

for image in data_path_folder:
    if image[-1] == 'g':
        img = cv2.imread(data_path + "/" + image, cv2.IMREAD_GRAYSCALE)
        img_resize = cv2.resize(img,(640,360))
        X_data.append(img_resize)

img_data = np.array(X_data)
img_data = img_data.astype('float32')
#img_data /=255
img_data = np.array(img_data, dtype=np.uint8)
print(img_data.shape)


# In[ ]:


# create video file
out = cv2.VideoWriter('deer.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, (640,360), isColor=False) # used MJPG due to using OSX 
 
for i in range(len(img_data)):
    out.write(img_data[i])
out.release()

