# CropCure
🌾 A Django-based web application that detects crop diseases from leaf images using a trained CNN model, and provides disease details, symptoms, and treatments. Includes user authentication, multilingual support, and prediction history tracking.

## 🚀 Features

- 🖼️ Image Upload: Upload plant leaf images
- 🤖 CNN Model Integration: Predict diseases using a trained deep learning model
- 💾 User History: Saves prediction history per user using Django ORM
- 🌍 Multilingual Support: Integrated Google Translate for multi-language user experience
- 🔐 User Authentication: Sign-up, login, and user-specific data
- 🧠 Disease Knowledge Base: Displays disease symptoms,treatment,sample images

---

## 🛠️ Tech Stack

- **Backend**: Django (Python)
- **Frontend**: HTML, CSS, Bootstrap
- **Model**: Convolutional Neural Network (Keras / TensorFlow)
- **Database**: SQLite3 (default Django DB)
- **Tools**: Google Colab, Google Drive

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Create a virtual environment
python -m venv env
source env/bin/activate     # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Start the server
python manage.py runserver


## 📸 Output Screenshots

### 🔍 prediction images
![Model Output](images/Screenshot (109).png)
![Model Output](images/Screenshot (115).png)
![Model Output](images/Screenshot (116).png)

### 📊 Detection History
![History Page](images/Screenshot (117).png)



