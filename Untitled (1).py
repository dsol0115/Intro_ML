#!/usr/bin/env python
# coding: utf-8

# In[17]:


import cv2, sys, re


# In[18]:


sys.argv = ['untitled.py','./pic3.jpg']
if len(sys.argv) <=1:
    print("not input file")
    quit()
image_file = sys.argv[1]


# In[19]:


output_file = re.sub(r'\,jpg|jpeg|PNG$', '-mosaic.jpg',image_file)
mosaic_rate =40


# In[20]:


cascade_file = "haarcascade_frontalface_alt.xml"


# In[21]:


image = cv2.imread(image_file)
image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# In[22]:


cascade = cv2.CascadeClassifier(cascade_file)
face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=1, minSize=(100,100))

if len(face_list)==0:
    print("no face")
    quit()


# In[23]:


print(face_list)
color = (0,0,255)

for(x,y,w,h) in face_list:
    face_img = image[y:y+h, x:x+w]
    
    face_img = cv2.resize(face_img, (w//mosaic_rate, h//mosaic_rate))
    
    face_img = cv2.resize(face_img, (w,h),
                         interpolation=cv2.INTER_AREA)
    
    image[y:y+h, x:x+w] = face_img
    


# In[24]:


cv2.imwrite(output_file, image)


# In[ ]:




