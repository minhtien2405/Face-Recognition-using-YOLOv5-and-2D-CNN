# FACE RECOGNITION USING YOLOV5-FACE AND 2D CONVOLUTIONAL NEURAL NETWORK 

## INTRODUCTION

Facial recognition is a crucial  aspect of computer vision  research, with security,  surveillance, and human-computer  interaction applications. This  project aims to develop a real-time  and accurate face recognition  system by integrating YOLOv5-face  and a custom CNN model. ![ref1]

## PROJECT WORKFLOW

**1.Data Collection**: We gather a diverse dataset of images containing faces to train our models.

**2.Face detection**: We train the YOLOv5 model using preprocessed data to detect faces in images and videos.

**3.CNN Model Training**: Next, we build and train a custom CNN model on top of the YOLOv5 output to perform face recognition and identification.

**4.Real-time Face recognition**: We use OpenCV to render a real-time video after facial recognition and labeling.

## DATASET 
- The dataset consists of self-captured facial images  using a custom Python tool for image capture. 
- Images were taken under various lighting conditions,  backgrounds, and facial poses to ensure diversity.
- 
![](Aspose.Words.681f8292-1d65-4545-804d-61b761c962c0.010.jpeg)

![](Aspose.Words.681f8292-1d65-4545-804d-61b761c962c0.012.png)![ref5]

## METHODOLOGY

**YOLOv5 – Face Detection**

![](Aspose.Words.681f8292-1d65-4545-804d-61b761c962c0.014.jpeg)

YOLOV5 is a state-of-the-art  We adapt YOLOV5 for face object detection model known  detection by fine-tuning it on a  for its speed and accuracy. face detection dataset. 

Fine-tuning involves training  The trained YOLOV5-Face the model on annotated face  model can efficiently and  images to learn to detect faces  accurately detect faces in real- in various poses and lighting  time, making it suitable for real- conditions. world applications.


**2D CONVOLUTIONAL NERAL NETWORK**

![](Aspose.Words.681f8292-1d65-4545-804d-61b761c962c0.021.png)

A 2D Convolutional     Neural Network is a     deep learning model designed for processing visual data, such as     images.


It consists of convolutional layers that learn hierarchical patterns and features from the input images.

Each convolutional    layer uses filters to    extract different       features, such as      edges, textures, and shapes, from the input image.

The output of each     convolutional layer     goes through activation functions to introduce non-linearity and make the model capable of learning complex       patterns.

Pooling layers reduce the spatial dimensions of the feature maps, decreasing computational complexity while retaining essential features.

Fully connected layers further      process the       features and make predictions based on the learned     representations.
## RESULT
**Face detection**

- We use YOLOv5 to detect faces on images and return bounding boxes and coordinates of 5 facial key points,  which can be used for face alignment. 
- After that, we crop the face and save it in the  database for 2D CNN model training. 
![](Aspose.Words.681f8292-1d65-4545-804d-61b761c962c0.028.jpeg)

**2D CNN Model for Face Recognition![ref6]**

- **Architecture**: Our CNN consists of multiple convolutional layers followed by fully connected layers to learn features from face images.

![](Aspose.Words.681f8292-1d65-4545-804d-61b761c962c0.029.png)

- **Training**: We train the CNN on a separate dataset containing labeled face images for  identification. 
- **Face Embeddings**: The CNN learns to map each face into a high-dimensional vector  space, often referred to as face embeddings. 
- **Accuracy:** CNN model achieve 96% in the validation dataset ![](Aspose.Words.681f8292-1d65-4545-804d-61b761c962c0.030.jpeg)![](Aspose.Words.681f8292-1d65-4545-804d-61b761c962c0.031.jpeg)![ref1]

**Real-time face recognition![ref6]**

- The real-time facial recognition system successfully detects and recognizes faces in a live video stream. 
![](Aspose.Words.681f8292-1d65-4545-804d-61b761c962c0.032.jpeg)
- The system demonstrates fast and efficient performance,  providing real-time results without noticeable delays. ![](Aspose.Words.681f8292-1d65-4545-804d-61b761c962c0.033.jpeg)



## FUTURE IMPROVEMENTS

While our face recognition system is already impressive, there are always opportunities for enhancement:

•**Data Augmentation**: Expanding the training dataset using techniques like image rotation, flipping, and scaling.

•**Ensemble Methods**: Combining multiple models to improve accuracy and robustness further.

•**Real-world Deployment**: Optimizing the system for deployment in real-world scenarios, such as surveillance and access control.!


## CONCLUSION

- In conclusion, our face recognition project using YOLOv5- face and a 2D CNN model opens up exciting possibilities in the realm of computer vision.
- The combination of real-time face detection and accurate recognition showcases the power of deep learning in solving complex tasks.![ref1]

![](Aspose.Words.681f8292-1d65-4545-804d-61b761c962c0.035.png)

![](Aspose.Words.681f8292-1d65-4545-804d-61b761c962c0.036.png)
