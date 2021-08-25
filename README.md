# Face Mask Detector
This face mask detection program detects faces using a webcam and applies a machine learning model to determine whether or not the person is wearing a mask. It can be used to ensure people entering a classroom or family gathering are wearing a mask by simply setting it up near the entrance. A text-to-speech feature is also implemented in this program to remind people who are not wearing a mask to wear a mask. 
 
## Dataset
To train the machine learning model, we have used a [dataset](https://www.kaggle.com/dataset/9737b9841a2073b4872bea55968787b6a5b7c44c0f3f2b5c6ba008435c2f2cdf) from the Kaggle platform.

## The Machine Learning Model
As we were working with image data, the best tool for this is a [Convolutional Neural Network](https://searchenterpriseai.techtarget.com/definition/convolutional-neural-network) (CNN). This deep learning model can find patterns in the data in order to classify images. More details on training this model can be viewed in this file: [face_mask_detector.ipynb](https://github.com/thuvaragan25/Face-Mask-Detector/blob/main/Machine%20Learning%20Model/face_mask_detector.ipynb).
 
## Main Tools Used
- **Python** - the coding language that was used to make this project
- **Keras** - the deep learning framework that is used to train the machine learning model
- **OpenCV** - the library that was used to process images when training the model and also to access the webcam to detect faces

## How This works
<table>
  <tr>
    <td><video src="https://user-images.githubusercontent.com/79026921/130805155-60669a33-4f3f-4fea-8dde-bc9fee0fdebc.mp4"></td>
    <td><video src="https://user-images.githubusercontent.com/79026921/130805071-990a9b5b-c234-48a0-98dd-6807d5390fdb.mp4"></td>
  </tr>
 </table>

## How To Use
1. First download the repository
2. Then install requirements.txt to download all the required modules to run this program
```
pip install -r requirements.txt
```
3. Run `face_mask_detection.py`
4. You should now be able to use this program successfully

## Contributors
- [@thuvaragan25](https://github.com/thuvaragan25) - Worked on implementing the Convolutional Neural Network and training the machine learning model
- [@mdola19](https://github.com/mdola19) - Worked on detecting faces in live webcam feed to look for face masks using the machine learning model and displaying results