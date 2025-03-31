import os
import time
import cv2
import numpy as np
import tensorflow as tf
import base64
import ctypes
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS  # Enable CORS for frontend communication
import struct
import pefile
import time
from pymongo import MongoClient
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from cryptography.fernet import Fernet


if not os.path.exists("secret.key"):
    raise FileNotFoundError("Error: secret.key file not found!")

with open("secret.key", "rb") as key_file:
    secret_key = key_file.read()

cipher = Fernet(secret_key)


MONGO_URI = "mongodb+srv://buragaravi:qzlCHauz9boCgeCK@cluster0.aow0j7e.mongodb.net/"

try:
    client = MongoClient(MONGO_URI)
    db = client.get_database("Cluster0")
    # Check connection
    print("Connected to MongoDB successfully!")
    client.server_info()  # This will raise an exception if the connection fails
except Exception as e:
    print("Failed to connect to MongoDB:", e)
 
 # Encrypt function
def encrypt_data(data):
    return cipher.encrypt(data.encode()).decode()

# Decrypt function
def decrypt_data(data):
    return cipher.decrypt(data.encode()).decode()

print("Python Architecture:", struct.calcsize("P") * 8, "bit")

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Set JWT secret key
app.config["JWT_SECRET_KEY"] = "RaviBuraga"
jwt = JWTManager(app)

# Set Absolute Path for Model
MODEL_PATH = r"E:\\touch-to-type\\Model_1_testing_application\\backend\\initial_model_accurate.h5"

# Ensure Model File Exists Before Loading
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Load Trained Model
model = tf.keras.models.load_model(MODEL_PATH)

# Allowed File Types
ALLOWED_EXTENSIONS = {"bmp"}

# Function to Check File Format
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to Convert Image to Base64
def encode_image_to_base64(image):
    """
    Encodes a NumPy image array to a Base64 string.
    """
    _, buffer = cv2.imencode(".bmp", image)
    return base64.b64encode(buffer).decode("utf-8")

# Function to Preprocess Image for Model
def preprocess_image(image_array):
    """
    Preprocesses an image for the model.
    - Converts to grayscale (if not already)
    - Resizes to 224x224
    - Normalizes pixel values
    - Expands dimensions to match model input shape
    """
    img = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if len(image_array.shape) == 3 else image_array
    img = cv2.resize(img, (224, 224))  # Resize to model's input size
    img = img / 255.0  # Normalize pixel values

    # Expand dimensions to match model input shape (1, 224, 224, 1)
    img_array = np.expand_dims(img, axis=0)  # Add batch dimension
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension

    return img_array

# Function to Predict Blood Group and Confidence Score
def predict_blood_group(image_array):
    """
    Predicts blood group from an image using the trained model.
    - Returns predicted blood group and confidence score.
    """
    img_array = preprocess_image(image_array)
    prediction = model.predict(img_array)[0]  # Get first batch prediction

    predicted_label = np.argmax(prediction)  # Get class with highest probability
    confidence_score = float(prediction[predicted_label]) * 100  # Convert to percentage

    return map_label_to_blood_group(predicted_label), confidence_score

# Function to Map Model Output to Blood Group Labels
def map_label_to_blood_group(label):
    blood_groups = {
        0: "A+", 1: "A-", 2: "AB+", 3: "AB-",
        4: "B+", 5: "B-", 6: "O+", 7: "O-"
    }
    return blood_groups.get(label, "Unknown")

# Load SecuGen SDK DLL
SGFPLib = None
SCANNER_AVAILABLE = False

# Updated DLL path
dll_path = "C:\\Program Files\\SecuGen\\Drivers\\HU20AL\\sgfdu08ax64.dll"

try:
    if os.path.exists(dll_path):
        print("✅ DLL file exists.")
        try:
            SGFPLib = ctypes.WinDLL(dll_path)
            print("✅ DLL Loaded Successfully")
        except Exception as e:
            print(f"❌ Failed to load DLL: {e}")
    else:
        print("❌ DLL file NOT found. Check the path.")

    # Get list of functions
    if SGFPLib:
        functions = dir(SGFPLib)
        print("Available functions in DLL:", functions)

        # Define SGFPM_Create function
        if hasattr(SGFPLib, 'SGFPM_Create'):
            SGFPM_Create = SGFPLib.SGFPM_Create
            SGFPM_Create.restype = ctypes.c_int  # Return type is int
            SGFPM_Create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]  # Pointer argument

            # Create object
            h_sgfpm = ctypes.c_void_p()
            result = SGFPM_Create(ctypes.byref(h_sgfpm))

            if result == 0:
                print("✅ SecuGen Scanner Initialized Successfully!")
            else:
                print(f"❌ Error: Failed to initialize scanner. Code: {result}")

            # Define and call Init function
            if hasattr(SGFPLib, 'SGFPM_Init'):
                SGFPLib.SGFPM_Init.argtypes = []
                SGFPLib.SGFPM_Init.restype = ctypes.c_int
                h_result = SGFPLib.SGFPM_Init()
                print("SGFPM_Init Result:", h_result)

                SCANNER_AVAILABLE = True
                print("✅ SecuGen SDK loaded successfully.")
            else:
                print("❌ SGFPM_Init function not found in DLL.")
        else:
            print("❌ SGFPM_Create function not found in DLL.")
    else:
        print("❌ SGFPLib is None, skipping function calls.")

except Exception as e:
    print(f"⚠️ Warning: SecuGen SDK not found. Scanner feature will be disabled. Error: {e}")

# Capture fingerprint function
def capture_fingerprint_secugen():
    if not SCANNER_AVAILABLE:
        return None, "Error: SecuGen scanner not available."

    try:
        print("Initializing SecuGen Scanner...")

        h_result = SGFPLib.Init()
        if h_result != 0:
            print("❌ Error: Failed to initialize scanner.")
            return None, "Error: Failed to initialize scanner."

        print("🛑 Waiting for fingerprint scan... Place your finger on the scanner.")
        time.sleep(2)  # Give user time to place finger

        # Capture fingerprint image
        IMAGE_WIDTH = 320  # Adjust as per scanner specs
        IMAGE_HEIGHT = 480  # Adjust as per scanner specs
        img_buffer = np.zeros((IMAGE_WIDTH * IMAGE_HEIGHT,), dtype=np.uint8)

        SGFPLib.GetImage.argtypes = [ctypes.POINTER(ctypes.c_ubyte)]
        SGFPLib.GetImage.restype = ctypes.c_int

        h_result = SGFPLib.GetImage(img_buffer.ctypes.data)
        if h_result != 0:
            return None, "❌ Error: Failed to capture fingerprint image."

        # Reshape buffer into an image format
        fingerprint_image = img_buffer.reshape(IMAGE_HEIGHT, IMAGE_WIDTH)

        return fingerprint_image, None

    except Exception as e:
        return None, f"❌ Error: {str(e)}"

@app.route("/upload-single", methods=["POST"])
def upload_single():
    """
    Handles single BMP image upload for blood group detection (without storing it).
    - Returns predicted blood group & confidence score.
    """
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]

    # Check if file is empty
    if file.filename == "":
        return jsonify({"error": "Empty file uploaded"}), 400

    # Validate file type
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Only BMP images are allowed."}), 400

    try:
        # Read image directly from memory
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Validate if image was read successfully
        if image is None:
            return jsonify({"error": "Error reading image. Please upload a valid BMP file."}), 400

        # Predict blood group and confidence score
        result, confidence = predict_blood_group(image)

        # Convert input image to Base64
        image_base64 = encode_image_to_base64(image)

        return jsonify({
            "input_image": image_base64,
            "result": result,
            "confidence": round(confidence, 2)  # Round to 2 decimal places
        })

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# API: Capture Fingerprint Using SecuGen Scanner & Predict Blood Group
@app.route("/scan-fingerprint-secugen", methods=["POST"])
def scan_fingerprint_secugen():
    """
    Captures a fingerprint from the SecuGen scanner and predicts the blood group.
    """
    image, error = capture_fingerprint_secugen()

    if error:
        return jsonify({"error": error}), 400

    result, confidence = predict_blood_group(image)

    # Convert fingerprint image to Base64
    image_base64 = encode_image_to_base64(image)

    return jsonify({
        "input_image": image_base64,
        "result": result,
        "confidence": round(confidence, 2)  # Round to 2 decimal places
    })

# API: Capture Fingerprint Using Webcam (Fallback)
@app.route("/scan-fingerprint", methods=["POST"])
def scan_fingerprint():
    """
    Handles automatic fingerprint scanning via webcam (fallback option).
    - Captures a fingerprint image
    - Predicts blood group & confidence score
    - Returns both the input image (Base64) and the prediction
    """
    cap = cv2.VideoCapture(0)  # 0 = First detected camera

    if not cap.isOpened():
        return jsonify({"error": "Error: Could not open webcam."}), 400

    print("Capturing image... Place your finger on the scanner.")
    time.sleep(2)  # Give user time to adjust their finger
    ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()

    if not ret:
        return jsonify({"error": "Error: Failed to capture fingerprint image."}), 400

    result, confidence = predict_blood_group(frame)

    # Convert captured image to Base64
    image_base64 = encode_image_to_base64(frame)

    return jsonify({
        "input_image": image_base64,
        "result": result,
        "confidence": round(confidence, 2)  # Round to 2 decimal places
    })
    
    
    @app.route('/register', methods=['POST'])
    def register():
        data = request.json
        username = encrypt_data(data['username'])
        email = encrypt_data(data['email'])
        password = encrypt_data(data['password'])

    # Check if user already exists
        if collection.find_one({"email": email}):
            return jsonify({"message": "User already registered"}), 400

        user_data = {"username": username, "email": email, "password": password}
        collection.insert_one(user_data)
        return jsonify({"message": "User registered successfully"}), 201

# User Login
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = encrypt_data(data['email'])
    password = encrypt_data(data['password'])

    user = collection.find_one({"email": email, "password": password})
    if user:
        return jsonify({"message": "Login successful"}), 200
    return jsonify({"message": "Invalid credentials. Please register."}), 401

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)