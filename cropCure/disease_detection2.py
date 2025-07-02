from keras.models import load_model
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

# Define class names and mappings
class_name = [
    'Apple___Black_rot', 'Apple___healthy', 'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___healthy', 'Potato___Late_blight', 'Raspberry___healthy',
    'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___healthy'
]

class_mapping = {
    'Apple___Black_rot': 'Apple Black Rot',
    'Apple___healthy': 'Apple Healthy',
    'Blueberry___healthy': 'Blueberry Healthy',
    'Cherry_(including_sour)___Powdery_mildew': 'Cherry Powdery Mildew',
    'Cherry_(including_sour)___healthy': 'Cherry Healthy',
    'Corn_(maize)___Common_rust_': 'Corn Common Rust',
    'Corn_(maize)___Northern_Leaf_Blight': 'Corn Northern Leaf Blight',
    'Corn_(maize)___healthy': 'Corn Healthy',
    'Grape___Black_rot': 'Grape Black Rot',
    'Grape___Esca_(Black_Measles)': 'Grape Esca',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Grape Leaf Blight',
    'Grape___healthy': 'Grape Healthy',
    'Orange___Haunglongbing_(Citrus_greening)': 'Orange Huanglongbing',
    'Peach___Bacterial_spot': 'Peach Bacterial Spot',
    'Peach___healthy': 'Peach Healthy',
    'Pepper,_bell___healthy': 'Pepper Bell Healthy',
    'Potato___Late_blight': 'Potato Late Blight',
    'Raspberry___healthy': 'Raspberry Healthy',
    'Soybean___healthy': 'Soybean Healthy',
    'Squash___Powdery_mildew': 'Squash Powdery Mildew',
    'Strawberry___Leaf_scorch': 'Strawberry Leaf Scorch',
    'Strawberry___healthy': 'Strawberry Healthy',
    'Tomato___Leaf_Mold': 'Tomato Leaf Mold',
    'Tomato___Septoria_leaf_spot': 'Tomato Septoria Leaf Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Tomato Yellow Leaf Curl Virus',
    'Tomato___healthy': 'Tomato Healthy'
}

# Dictionary for Symptoms and Treatment
disease_details = {
    "Apple Black Rot": {
        "symptoms": "Dark brown or black circular spots on leaves and fruit, leading to decay.",
        "treatment": "Remove infected fruits, prune affected areas, apply fungicides like Captan or Mancozeb."
    },
    "Apple Healthy": {
        "symptoms": "No visible disease symptoms. Leaves and fruit are healthy and vibrant.",
        "treatment": "Maintain regular watering, fertilization, and pruning to keep the plant healthy."
    },
    "Blueberry Healthy": {
        "symptoms": "No signs of disease. Leaves are green, and berries develop properly.",
        "treatment": "Provide well-drained soil and adequate sunlight for continued health."
    },
    "Cherry Powdery Mildew": {
        "symptoms": "White powdery fungal growth on leaves, shoots, and fruit.",
        "treatment": "Apply sulfur-based fungicides, remove affected leaves, and improve air circulation."
    },
    "Cherry Healthy": {
        "symptoms": "No signs of infection. Leaves and fruit are normal.",
        "treatment": "Regular pruning and proper watering will keep it disease-free."
    },
    "Corn Common Rust": {
        "symptoms": "Reddish-brown pustules on leaves, reducing photosynthesis.",
        "treatment": "Plant resistant varieties, rotate crops, and apply fungicides like Azoxystrobin."
    },
    "Corn Northern Leaf Blight": {
        "symptoms": "Long gray-green lesions on leaves that turn brown and spread quickly.",
        "treatment": "Use resistant hybrids, apply fungicides, and remove crop debris."
    },
    "Corn Healthy": {
        "symptoms": "Leaves and cobs are developing properly with no signs of disease.",
        "treatment": "Regular watering and fertilization help maintain plant health."
    },
    "Grape Black Rot": {
        "symptoms": "Dark brown circular spots on leaves, fruit shriveling into hard mummies.",
        "treatment": "Remove infected fruit and leaves, apply fungicides like Myclobutanil."
    },
    "Grape Esca": {
        "symptoms": "Brown streaks on leaves, wilting, and drying of grape clusters.",
        "treatment": "Prune infected vines, avoid overwatering, apply fungicides like Thiophanate-methyl."
    },
    "Grape Leaf Blight": {
        "symptoms": "Brown or black irregular spots on leaves, leading to premature defoliation.",
        "treatment": "Apply copper-based fungicides, remove infected leaves."
    },
    "Grape Healthy": {
        "symptoms": "No symptoms. Leaves and grapes are growing normally.",
        "treatment": "Regular pruning and disease monitoring will maintain health."
    },
    "Orange Huanglongbing": {
        "symptoms": "Yellowing leaves, small deformed fruit, bitter taste.",
        "treatment": "Remove infected trees, control psyllid insects, and use disease-free plants."
    },
    "Peach Bacterial Spot": {
        "symptoms": "Small dark water-soaked spots on leaves and fruit that enlarge over time.",
        "treatment": "Apply copper-based sprays, remove infected branches."
    },
    "Peach Healthy": {
        "symptoms": "Leaves and fruit are normal with no visible disease.",
        "treatment": "Maintain good orchard sanitation and proper watering."
    },
    "Pepper Bell Healthy": {
        "symptoms": "No symptoms. Leaves are green and fruit develops normally.",
        "treatment": "Regular watering and balanced fertilization help maintain health."
    },
    "Potato Late Blight": {
        "symptoms": "Dark brown patches on leaves and stems, rapid decay of tubers.",
        "treatment": "Use resistant varieties, apply copper-based fungicides, avoid overhead watering."
    },
    "Raspberry Healthy": {
        "symptoms": "No disease symptoms. Leaves are green and fruit develops properly.",
        "treatment": "Regular watering and pruning help maintain plant health."
    },
    "Soybean Healthy": {
        "symptoms": "No disease symptoms. Leaves and pods are growing normally.",
        "treatment": "Ensure good crop rotation and adequate soil nutrition."
    },
    "Squash Powdery Mildew": {
        "symptoms": "White powdery fungal growth on leaves, leading to yellowing and drying.",
        "treatment": "Apply sulfur-based fungicides, remove infected leaves."
    },
    "Strawberry Leaf Scorch": {
        "symptoms": "Brown or reddish edges on leaves, leading to leaf drop.",
        "treatment": "Prune infected leaves, improve air circulation, apply fungicides."
    },
    "Strawberry Healthy": {
        "symptoms": "Leaves and fruit are growing properly with no disease symptoms.",
        "treatment": "Regular watering and balanced fertilizers will maintain plant health."
    },
    "Tomato Leaf Mold": {
        "symptoms": "Yellow spots on leaves with moldy gray patches underneath.",
        "treatment": "Apply fungicides, increase ventilation, and remove infected leaves."
    },
    "Tomato Septoria Leaf Spot": {
        "symptoms": "Small circular spots with dark centers and yellow halos on leaves.",
        "treatment": "Remove infected leaves, apply copper-based fungicides."
    },
    "Tomato Yellow Leaf Curl Virus": {
        "symptoms": "Yellowing and curling of leaves, stunted growth.",
        "treatment": "Use virus-free seedlings, remove infected plants, and control whiteflies."
    },
    "Tomato Healthy": {
        "symptoms": "No visible disease. Leaves and fruit are normal.",
        "treatment": "Regular watering, fertilization, and pruning keep plants healthy."
    }
}

   


def detect_disease(image_path):
    # Load the trained model
    cnn = tf.keras.models.Sequential()

    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=[128, 128, 3]))
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
    cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
    cnn.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(units=26, activation='softmax'))
    cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    cnn.load_weights('cropCure/static/final_model82.h5')
    print("Model weights loaded successfully.")

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])

    predictions = cnn.predict(input_arr)
    result_index = np.argmax(predictions)
    confidence = round(float(np.max(predictions)) * 100, 2)  # Get highest probability (confidence level)

    model_prediction = class_name[result_index]
    disease_name = class_mapping[model_prediction]
    res_symptoms = disease_details[disease_name]["symptoms"]
    res_treatment = disease_details[disease_name]["treatment"]

    # print(res_symptoms)
    # print(res_treatment)
    result = {
        "disease": disease_name,
        "confidence": f"{confidence}%",
        "symptoms": res_symptoms,
        "treatment": res_treatment
    }
    # print("disease_detect called")
    # print(result)
    return result
'''
from keras.models import load_model
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
from collections import Counter

# Load pre-trained models
model_color = load_model('C:/Users/rakhi/Downloads/model_color1.h5')
model_gray = load_model('C:/Users/rakhi/Downloads/model_gray1.h5')
model_segment = load_model('C:/Users/rakhi/Downloads/model_segment1.h5')

# Define your class names (from your dataset's labels)
class_name = [
    'Apple__Apple_scab','Apple___Black_rot', 'Apple__Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',  'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy', 'Potato___Early_blight','Potato___Late_blight', 'Potato___healthy','Raspberry___healthy',
    'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Mapping class name to readable format
class_mapping = {
    'Apple__Apple_scab': 'Apple Scab',
    'Apple___Black_rot': 'Apple Black Rot',
    'Apple__Cedar_apple_rust' : 'Cedar Apple Rust',
    'Apple___healthy': 'Apple Healthy',
    'Blueberry___healthy': 'Blueberry Healthy',
    'Cherry_(including_sour)___Powdery_mildew': 'Cherry Powdery Mildew',
    'Cherry_(including_sour)___healthy': 'Cherry Healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Corn Cercospora Leaf Spot',
    'Corn_(maize)___Common_rust_': 'Corn Common Rust',
    'Corn_(maize)___Northern_Leaf_Blight': 'Corn Northern Leaf Blight',
    'Corn_(maize)___healthy': 'Corn Healthy',
    'Grape___Black_rot': 'Grape Black Rot',
    'Grape___Esca_(Black_Measles)': 'Grape Esca',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Grape Leaf Blight',
    'Grape___healthy': 'Grape Healthy',
    'Orange___Haunglongbing_(Citrus_greening)': 'Orange Huanglongbing',
    'Peach___Bacterial_spot': 'Peach Bacterial Spot',
    'Peach___healthy': 'Peach Healthy',
    'Pepper,_bell___Bacterial_spot': 'Pepper Bell Bacterial Spot',
    'Pepper,_bell___healthy': 'Pepper Bell Healthy',
    'Potato___Early_blight': 'Potato Early Blight',
    'Potato___Late_blight': 'Potato Late Blight',
    'Potato___healthy': 'Potato Healthy',
    'Raspberry___healthy': 'Raspberry Healthy',
    'Soybean___healthy': 'Soybean Healthy',
    'Squash___Powdery_mildew': 'Squash Powdery Mildew',
    'Strawberry___Leaf_scorch': 'Strawberry Leaf Scorch',
    'Strawberry___healthy': 'Strawberry Healthy',
    'Tomato___Bacterial_spot': 'Tomato Bacterial Spot',
    'Tomato___Early_blight': 'Tomato Early Blight',
    'Tomato___Late_blight': 'Tomato Late Blight',
    'Tomato___Leaf_Mold': 'Tomato Leaf Mold',
    'Tomato___Septoria_leaf_spot': 'Tomato Septoria Leaf Spot',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Tomato Spider Mites',
    'Tomato___Target_Spot': 'Tomato Target Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Tomato Yellow Leaf Curl Virus',
    'Tomato___Tomato_mosaic_virus': 'Tomato Mosaic Virus',
    'Tomato___healthy': 'Tomato Healthy'
}

# Disease details (symptoms and treatment info)
disease_details = {
    "Apple Black Rot": {
        "symptoms": "Dark brown or black circular spots on leaves and fruit, leading to decay.",
        "treatment": "Remove infected fruits, prune affected areas, apply fungicides like Captan or Mancozeb."
    },
    "Apple Healthy": {
        "symptoms": "No visible disease symptoms. Leaves and fruit are healthy and vibrant.",
        "treatment": "Maintain regular watering, fertilization, and pruning to keep the plant healthy."
    },
    "Blueberry Healthy": {
        "symptoms": "No signs of disease. Leaves are green, and berries develop properly.",
        "treatment": "Provide well-drained soil and adequate sunlight for continued health."
    },
    "Cherry Powdery Mildew": {
        "symptoms": "White powdery fungal growth on leaves, shoots, and fruit.",
        "treatment": "Apply sulfur-based fungicides, remove affected leaves, and improve air circulation."
    },
    "Cherry Healthy": {
        "symptoms": "No signs of infection. Leaves and fruit are normal.",
        "treatment": "Regular pruning and proper watering will keep it disease-free."
    },
    "Corn Common Rust": {
        "symptoms": "Reddish-brown pustules on leaves, reducing photosynthesis.",
        "treatment": "Plant resistant varieties, rotate crops, and apply fungicides like Azoxystrobin."
    },
    "Corn Northern Leaf Blight": {
        "symptoms": "Long gray-green lesions on leaves that turn brown and spread quickly.",
        "treatment": "Use resistant hybrids, apply fungicides, and remove crop debris."
    },
    "Corn Healthy": {
        "symptoms": "Leaves and cobs are developing properly with no signs of disease.",
        "treatment": "Regular watering and fertilization help maintain plant health."
    },
    "Grape Black Rot": {
        "symptoms": "Dark brown circular spots on leaves, fruit shriveling into hard mummies.",
        "treatment": "Remove infected fruit and leaves, apply fungicides like Myclobutanil."
    },
    "Grape Esca": {
        "symptoms": "Brown streaks on leaves, wilting, and drying of grape clusters.",
        "treatment": "Prune infected vines, avoid overwatering, apply fungicides like Thiophanate-methyl."
    },
    "Grape Leaf Blight": {
        "symptoms": "Brown or black irregular spots on leaves, leading to premature defoliation.",
        "treatment": "Apply copper-based fungicides, remove infected leaves."
    },
    "Grape Healthy": {
        "symptoms": "No symptoms. Leaves and grapes are growing normally.",
        "treatment": "Regular pruning and disease monitoring will maintain health."
    },
    "Orange Huanglongbing": {
        "symptoms": "Yellowing leaves, small deformed fruit, bitter taste.",
        "treatment": "Remove infected trees, control psyllid insects, and use disease-free plants."
    },
    "Peach Bacterial Spot": {
        "symptoms": "Small dark water-soaked spots on leaves and fruit that enlarge over time.",
        "treatment": "Apply copper-based sprays, remove infected branches."
    },
    "Peach Healthy": {
        "symptoms": "Leaves and fruit are normal with no visible disease.",
        "treatment": "Maintain good orchard sanitation and proper watering."
    },
    "Pepper Bell Healthy": {
        "symptoms": "No symptoms. Leaves are green and fruit develops normally.",
        "treatment": "Regular watering and balanced fertilization help maintain health."
    },
    "Potato Late Blight": {
        "symptoms": "Dark brown patches on leaves and stems, rapid decay of tubers.",
        "treatment": "Use resistant varieties, apply copper-based fungicides, avoid overhead watering."
    },
    "Raspberry Healthy": {
        "symptoms": "No disease symptoms. Leaves are green and fruit develops properly.",
        "treatment": "Regular watering and pruning help maintain plant health."
    },
    "Soybean Healthy": {
        "symptoms": "No disease symptoms. Leaves and pods are growing normally.",
        "treatment": "Ensure good crop rotation and adequate soil nutrition."
    },
    "Squash Powdery Mildew": {
        "symptoms": "White powdery fungal growth on leaves, leading to yellowing and drying.",
        "treatment": "Apply sulfur-based fungicides, remove infected leaves."
    },
    "Strawberry Leaf Scorch": {
        "symptoms": "Brown or reddish edges on leaves, leading to leaf drop.",
        "treatment": "Prune infected leaves, improve air circulation, apply fungicides."
    },
    "Strawberry Healthy": {
        "symptoms": "Leaves and fruit are growing properly with no disease symptoms.",
        "treatment": "Regular watering and balanced fertilizers will maintain plant health."
    },
    "Tomato Leaf Mold": {
        "symptoms": "Yellow spots on leaves with moldy gray patches underneath.",
        "treatment": "Apply fungicides, increase ventilation, and remove infected leaves."
    },
    "Tomato Septoria Leaf Spot": {
        "symptoms": "Small circular spots with dark centers and yellow halos on leaves.",
        "treatment": "Remove infected leaves, apply copper-based fungicides."
    },
    "Tomato Yellow Leaf Curl Virus": {
        "symptoms": "Yellowing and curling of leaves, stunted growth.",
        "treatment": "Use virus-free seedlings, remove infected plants, and control whiteflies."
    },
    "Tomato Healthy": {
        "symptoms": "No visible disease. Leaves and fruit are normal.",
        "treatment": "Regular watering, fertilization, and pruning keep plants healthy."
    }
}

# Model weights (based on F1-score or accuracy of each model)
model_weights = {
    "color": 0.40,
    "gray": 0.25,
    "segment": 0.35
}

def preprocess_image(image_path, mode="color"):
    image = cv2.imread(image_path)

    # Check if image loaded correctly
    if image is None:
        print(f"Error: Could not load image from {image_path}. Check file path and permissions.")
        return None  # or raise an exception

    if mode == "gray":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (128, 128))
        image = np.expand_dims(image, axis=-1)  # for channels
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128, 128))

    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def detect_disease(image_path):
    weighted_scores = {}

    # Color model prediction
    img_color = preprocess_image(image_path, mode="color")
    if img_color is None:  # Handle image loading error
        return {"error": "Could not load image"}

    pred_color = model_color.predict(img_color)
    
    idx_color = np.argmax(pred_color)
    disease_color = class_name[idx_color]
    print(disease_color)
    confidence_color = np.max(pred_color)
    weighted_scores[disease_color] = weighted_scores.get(disease_color, 0) + confidence_color * model_weights["color"]

    # Grayscale model prediction
    img_gray = preprocess_image(image_path, mode="gray")
    pred_gray = model_gray.predict(img_gray)
    idx_gray = np.argmax(pred_gray)
    disease_gray = class_name[idx_gray]
    print(disease_gray)
    confidence_gray = np.max(pred_gray)
    weighted_scores[disease_gray] = weighted_scores.get(disease_gray, 0) + confidence_gray * model_weights["gray"]

    # Segmented model prediction
    img_segment = preprocess_image(image_path, mode="color")
    pred_segment = model_segment.predict(img_segment)
    idx_segment = np.argmax(pred_segment)
    disease_segment = class_name[idx_segment]
    print(disease_segment)
    confidence_segment = np.max(pred_segment)
    weighted_scores[disease_segment] = weighted_scores.get(disease_segment, 0) + confidence_segment * model_weights["segment"]

    # Final prediction based on highest weighted score
    final_disease = max(weighted_scores, key=weighted_scores.get)
    final_confidence = round(weighted_scores[final_disease] * 100, 2)

    readable_name = class_mapping[final_disease]
    res_symptoms = disease_details[readable_name]["symptoms"]
    res_treatment = disease_details[readable_name]["treatment"]

    result = {
        "disease": readable_name,
        "confidence": f"{final_confidence}%",
        "symptoms": res_symptoms,
        "treatment": res_treatment
    }

    return result
    '''     