import streamlit as st
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from keras.models import Model, load_model
import pickle

st.header("Van Gogh paintings Identifier")

no_of_patches = 20

def get_patch_feature_vgg19(patch):
    
    patch_input = np.expand_dims(patch, axis = 0)             #add an extra dimension for batch
    patch_preprocessed_input = preprocess_input(patch_input)               

    p_feature = vgg19model.predict(patch_preprocessed_input) 
    p_feature = p_feature.reshape(25088)

    return p_feature


def agg_pred_mean(pred):
  arr_pos = []
  arr_neg = []

  for predItem in pred:
    if(predItem >= 0):
      arr_pos.append(predItem)
    else:
      arr_neg.append(predItem)


  avg_pos = np.mean(arr_pos) if(len(arr_pos) > 0) else 0
  
  avg_neg = np.abs(np.mean(arr_neg)) if(len(arr_neg) > 0) else 0

  cl = 1 if(avg_pos > avg_neg) else 0
  
  return cl,arr_pos,arr_neg

patch_size = 224                            #patch_size for vgg19 model

uploaded_file = st.file_uploader("Choose a image file", type="png")

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    # Now do something with the image! For example, let's display it:
    st.image(resized, channels="RGB")
    
    genrate_pred = st.button("Generate Prediction")   
    
    if genrate_pred:
        
        filename = '/content/gdrive/MyDrive/Colab Notebooks/29. Identification of Van Gogh paintings/vgdb_2016/vgdb_2016/final_model_svm'
        clf = pickle.load(open(filename, 'rb'))
        
        vgg19model = VGG19(include_top = False, weights = 'imagenet')
        
        patches = []
        
        image_height = opencv_image.shape[0]
        image_width = opencv_image.shape[1]
        #Subtract patch_size from image's height and width to avoid out of bounds error
        range_x = image_height - patch_size
        range_y = image_width - patch_size
        
        #Generate patches for each image. The number of patches are passed as parameter.
        for i in range(no_of_patches):
                
            #Generate patch from random area of the image
            x = np.random.randint(low = 0, high = range_x)
            y = np.random.randint(low = 0, high = range_y)
        
            #The patch is calculated by adding the patch_size to both x and y co-ordinates
            patch = opencv_image[x : x+patch_size, y : y+patch_size, :]
            patches.append(patch)
        
        predictions = []
        patch_pred = []
        
        for patch in patches:
            
            patch_feature = get_patch_feature_vgg19(patch)
            
            pred_proba = clf.decision_function([patch_feature])      
        
            patch_pred.append(pred_proba) 
        
        
        predictions.append(patch_pred)
        
        y_pred_mean = []
        classes = ['not Van Gogh','Van Gogh']
        
        clss,arr_pos, arr_neg = agg_pred_mean(predictions[0])
        
        pos_prob = (len(arr_pos)/no_of_patches)
        neg_prob = (len(arr_neg)/no_of_patches)
        
        avg = 0
        
        if(pos_prob >= neg_prob):
            avg = pos_prob
            clss =1
        else:
            avg = neg_prob
            clss=0
            
        st.title("The artist of this image is {}".format(classes[clss]))
        st.write("The probability is {}%".format(int(avg * 100)))


