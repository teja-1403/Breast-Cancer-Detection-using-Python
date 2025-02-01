import cv2  # Import OpenCV library for image processing
import numpy as np  # Import NumPy for numerical operations
import os  # Import os library for working with the file system
from tqdm import tqdm  # Import tqdm for progress bar visualization
from PIL import Image  # Import Python Imaging Library for image handling
from sklearn.model_selection import train_test_split  # Import train_test_split from scikit-learn for data splitting
from sklearn.linear_model import LogisticRegression  # Import the Logistic Regression classifier from scikit-learn
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score  # Import evaluation metrics
import streamlit as st  # Install streamlit python package
from io import BytesIO  # Import BytesIO from the io module for handling byte data
import requests  # Import requests to send HTTP requests to web servers

# Function to load a dataset from a directory
def Dataset_loader(DIR, RESIZE):
    IMG = []
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    for IMAGE_NAME in tqdm(os.listdir(DIR)):
        PATH = os.path.join(DIR, IMAGE_NAME)
        _, ftype = os.path.splitext(PATH)
        if ftype == ".jpg":
            img = read(PATH)
            img = cv2.resize(img, (RESIZE, RESIZE))
            IMG.append(np.array(img))
    return IMG

# Load datasets (benign, malignant, normal and non_breast)
benign = np.array(Dataset_loader(r'C:\Users\pra\OneDrive - vitap.ac.in\Desktop\IGCAR PROJECT\Images\Benign', 64))
malignant = np.array(Dataset_loader(r'C:\Users\pra\OneDrive - vitap.ac.in\Desktop\IGCAR PROJECT\Images\Malignant', 64))
normal = np.array(Dataset_loader(r'C:\Users\pra\OneDrive - vitap.ac.in\Desktop\IGCAR PROJECT\Images\Normal', 64))
non_breast = np.array(Dataset_loader(r'C:\Users\pra\OneDrive - vitap.ac.in\Desktop\IGCAR PROJECT\Images\Non_breast', 64)) 

# Create labels for the datasets
benign_label = np.zeros(len(benign))
malignant_label = np.ones(len(malignant))
normal_label = 2 * np.ones(len(normal))
non_breast_label = 3 * np.ones(len(non_breast))

# Combine the datasets and labels
X = np.concatenate((benign, malignant, normal, non_breast), axis=0)
Y = np.concatenate((benign_label, malignant_label, normal_label, non_breast_label), axis=0)

# Flatten the images to use as feature vectors
X = X.reshape(X.shape[0], -1)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Initialize and train the Logistic Regression classifier 
lr_classifier = LogisticRegression(random_state=42, max_iter=1500)
lr_classifier.fit(X_train, Y_train)

# Evaluate the model on the test data
Y_pred = lr_classifier.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred, average='weighted')
recall = recall_score(Y_test, Y_pred, average='weighted')
precision = precision_score(Y_test, Y_pred, average='weighted')

# Evaluate performance measures
accuracy = round(accuracy * 100, 2)
f1 = round(f1 * 100, 2)
recall = round(recall * 100, 2)
precision = round(precision * 100, 2)

print("Accuracy:", accuracy, "%")
print("F1-score:", f1, "%")
print("Recall:", recall, "%")
print("Precision:", precision, "%")

# Add Streamlit UI elements
st.title("Breast Cancer Detection")

# Function to preprocess a user-provided test image
def preprocess_user_test_image(user_image):
    try:
        img = Image.open(user_image).convert("RGB")
        img = img.resize((64, 64))
        img = np.array(img)
        img = img.reshape(1, -1)  # Flatten the image
        return img
    except Exception as e:
        print("Could not process the image. Make sure it's a valid image file!")

# Function to preprocess an image from a URL
def preprocess_url_image(image_url):
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        img = img.resize((64, 64))
        img = np.array(img)
        img = img.reshape(1, -1)  # Flatten the image
        return img
    except Exception as e:
        print("Could not process the image from the URL. Please check the URL and try again.")

# Function to test a single user-provided image
def test_user_image(lr_classifier, user_image):

    # Preprocess the user-provided test image
    test_image = preprocess_user_test_image(user_image)

    if test_image is not None:
        # Reshape the image to a 2D array
        test_image = test_image.reshape(1, -1)

        # Predict the class of the test image
        predictions = lr_classifier.predict(test_image)

        # Define class labels
        class_labels = ['Benign', 'Malignant', 'Normal', 'None']
        predicted_class_label = class_labels[int(predictions)]

        return predicted_class_label
    else:
        return "This image cannot be classified as image segmentation failed."

# Function to test a single user-provided image
def test_user_url_image(lr_classifier, image_url):

    # Preprocess the user-provided url image
    test_image = preprocess_url_image(image_url)

    if test_image is not None:
        # Reshape the image to a 2D array
        test_image = test_image.reshape(1, -1)

        # Predict the class of the test image
        predictions = lr_classifier.predict(test_image)

        # Define class labels
        class_labels = ['Benign', 'Malignant', 'Normal', 'None']
        predicted_class_label = class_labels[int(predictions)]

        return predicted_class_label
    else:
        return "This image cannot be classified as image segmentation failed."

# Option to select an image from your device
st.header("Select an image from your device (.jpg only)")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg"], accept_multiple_files=False, key="device_upload")

# Button layout for file upload section
col1, col2 = st.columns(2)

if col2.button("Classify Image", key="button1"):
    if uploaded_file is not None:
        if uploaded_file.size > 200 * 1024 * 1024:  # Check if the file size is greater than 200MB
            st.warning("The uploaded image is too large (max 200MB). Please upload a smaller image.")
        else:
            result = test_user_image(lr_classifier, uploaded_file)
            st.image(uploaded_file, use_column_width=True)
            if result != 'None':
                st.markdown(f"<h1 style='color: red;'>Test image is classified as {result}</h1>", unsafe_allow_html=True)
                st.write("Accuracy: ", accuracy, "%")
                st.write("F1-score: ", f1, "%")
                st.write("Recall: ", recall, "%")
                st.write("Precision: ", precision, "%")
            else:
                st.markdown(f"<h1 style='color: red;'>Test image can't be classified</h1>", unsafe_allow_html=True)
                
# Button to clear the results
if col1.button("Clear", key="clear_button1"):
    st.empty()

# Option to enter the URL of an image from the web
st.header("Enter the URL of an image from the Web (.jpg only)")
image_url = st.text_input("Enter the URL of an image from the web")

# Button layout for URL input section
col3, col4 = st.columns(2)

if col4.button("Classify Image", key="button2"):
    if image_url:
        if not image_url.lower().endswith((".jpg", ".jpeg")):
            st.warning("Please enter a URL pointing to a .jpg or .jpeg image.")
        else:
            result = test_user_url_image(lr_classifier, image_url)
            st.image(preprocess_url_image(image_url), use_column_width=True)
            if result != 'None':
                st.markdown(f"<h1 style='color: red;'>Test image is classified as {result}</h1>", unsafe_allow_html=True)
                st.write("Accuracy: ", accuracy, "%")
                st.write("F1-score: ", f1, "%")
                st.write("Recall: ", recall, "%")
                st.write("Precision: ", precision, "%")
            else:
                st.markdown(f"<h1 style='color: red;'>Test image can't be classified</h1>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a valid URL before classifying.")

# Button to clear the results
if col3.button("Clear", key="clear_button2"):
    st.empty()
