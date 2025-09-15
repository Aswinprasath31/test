import streamlit as st
import os
import shutil
import yaml
from ultralytics import YOLO
import glob
import json
import subprocess

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="🚗 YOLOv8 Car Detection",
    page_icon="🚗",
    layout="wide"
)

# --- PATHS & CONFIG ---
KAGGLE_DATASET_ID = "ahmedhamdiph/car-detection-dataset"
WORKING_DIR = "working"
DOWNLOAD_PATH = os.path.join(WORKING_DIR, "kaggle_download")
SRC_DATASET_PATH = os.path.join(DOWNLOAD_PATH, "car_dataset-master") # The folder structure inside the zip
DST_DATASET_PATH = os.path.join(WORKING_DIR, "datasets/car-detection-dataset")
YAML_SAVE_PATH = os.path.join(WORKING_DIR, "data.yaml")
PROJECT_NAME = "car_detection_yolov8"

# Ensure the working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)


# --- HELPER FUNCTIONS ---
def setup_dataset_from_kaggle(kaggle_api_json):
    """Downloads dataset from Kaggle, prepares it, and creates the YAML file."""
    with st.spinner("Setting up Kaggle credentials and downloading dataset... This might take a few minutes."):
        # 1. Setup Kaggle credentials securely
        try:
            kaggle_dir = os.path.expanduser("~/.kaggle")
            os.makedirs(kaggle_dir, exist_ok=True)
            kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
            with open(kaggle_json_path, "w") as f:
                json.dump(kaggle_api_json, f)
            os.chmod(kaggle_json_path, 0o600) # Set secure permissions
        except Exception as e:
            st.error(f"Failed to set up Kaggle credentials: {e}")
            return False

        # 2. Download and unzip dataset using Kaggle CLI
        if os.path.exists(DOWNLOAD_PATH):
            shutil.rmtree(DOWNLOAD_PATH)
        os.makedirs(DOWNLOAD_PATH)

        try:
            command = [
                "kaggle", "datasets", "download",
                "-d", KAGGLE_DATASET_ID,
                "-p", DOWNLOAD_PATH,
                "--unzip"
            ]
            st.info(f"Running command: `{' '.join(command)}`")
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            st.text(result.stdout)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            st.error(f"Failed to download dataset from Kaggle. Make sure 'kaggle' is installed and in your system's PATH.")
            st.error(f"Error details: {e.stderr if isinstance(e, subprocess.CalledProcessError) else 'kaggle command not found.'}")
            return False

        # 3. Verify download and copy to working directory
        if not os.path.exists(SRC_DATASET_PATH):
            st.error(f"Dataset downloaded, but the expected folder '{SRC_DATASET_PATH}' was not found.")
            st.text(f"Contents of download directory for debugging: {os.listdir(DOWNLOAD_PATH)}")
            return False

        if os.path.exists(DST_DATASET_PATH):
            shutil.rmtree(DST_DATASET_PATH)
        shutil.copytree(SRC_DATASET_PATH, DST_DATASET_PATH)

        # 4. Create the data.yaml file for YOLO
        yaml_content = {
            'path': os.path.abspath(DST_DATASET_PATH),
            'train': 'train/images',
            'val': 'valid/images',
            'nc': 1,
            'names': ['car']
        }
        with open(YAML_SAVE_PATH, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

    st.session_state['dataset_ready'] = True
    st.success(f"✅ Dataset downloaded from Kaggle and prepared successfully.")
    return True

def find_latest_run_dir(project_name):
    """Finds the latest run directory within the project."""
    list_of_dirs = glob.glob(os.path.join('runs/detect', project_name + '*'))
    if not list_of_dirs:
        return None
    return max(list_of_dirs, key=os.path.getmtime)


# --- STREAMLIT APP LAYOUT ---
st.title("🚗 YOLOv8 Car Detection Trainer & Inspector")
st.markdown("This application downloads a car dataset from Kaggle, trains a YOLOv8 model, and runs inference.")

# --- Initialize Session State ---
if 'dataset_ready' not in st.session_state:
    st.session_state['dataset_ready'] = False
if 'training_complete' not in st.session_state:
    st.session_state['training_complete'] = False
if 'trained_model_path' not in st.session_state:
    st.session_state['trained_model_path'] = None

# --- SIDEBAR ---
st.sidebar.header("⚙️ Model & Training Configuration")
selected_model = st.sidebar.selectbox("Choose a YOLOv8 Model", ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt'])
epochs = st.sidebar.slider("Number of Epochs", 1, 100, 25)
img_size = st.sidebar.selectbox("Image Size (pixels)", [320, 640], index=1)
batch_size = st.sidebar.slider("Batch Size", 1, 32, 8)
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)


# --- MAIN WORKFLOW ---

# --- Step 1: Dataset Preparation ---
st.header("Step 1: Download & Prepare Dataset from Kaggle")
with st.expander("ℹ️ How to get your Kaggle API key"):
    st.markdown("""
    1.  Go to your Kaggle account settings: [https://www.kaggle.com/account](https://www.kaggle.com/account)
    2.  Scroll down to the **API** section.
    3.  Click `Create New API Token`. This will download the `kaggle.json` file.
    4.  Upload that file below. The app will handle the rest.
    """)

uploaded_file = st.file_uploader("Upload your `kaggle.json` file", type="json")

if st.button("Download & Prepare Dataset", disabled=(uploaded_file is None)):
    api_credentials = json.load(uploaded_file)
    setup_dataset_from_kaggle(api_credentials)

if st.session_state['dataset_ready']:
    st.success("Dataset is ready for training.")
    with st.expander("View `data.yaml` content"):
        with open(YAML_SAVE_PATH, 'r') as f:
            st.code(f.read(), language='yaml')

# --- Step 2: Model Training ---
st.header("Step 2: Train the YOLOv8 Model")
if st.button("Start Training", disabled=not st.session_state['dataset_ready']):
    with st.spinner(f"Training YOLOv8 model for {epochs} epochs... This can take a long time."):
        try:
            model = YOLO(selected_model)
            model.train(
                data=YAML_SAVE_PATH, epochs=epochs, imgsz=img_size, batch=batch_size,
                project='runs/detect', name=PROJECT_NAME, exist_ok=True
            )
            run_dir = find_latest_run_dir(PROJECT_NAME)
            best_weights_path = os.path.join(run_dir, "weights/best.pt")
            
            if os.path.exists(best_weights_path):
                st.session_state['training_complete'] = True
                st.session_state['trained_model_path'] = best_weights_path
                st.success(f"✅ Training complete! Model saved at: `{best_weights_path}`")
            else:
                st.error("Training finished, but 'best.pt' was not found.")
        except Exception as e:
            st.error(f"An error occurred during training: {e}")

# --- Step 3: Validation & Prediction ---
st.header("Step 3: Evaluate and Predict")
if st.button("Run Validation & Inference", disabled=not st.session_state.get('training_complete', False)):
    model_path = st.session_state.get('trained_model_path')
    if not model_path or not os.path.exists(model_path):
        st.error("Trained model not found. Please complete training first.")
    else:
        # --- Validation ---
        with st.spinner("Running validation..."):
            st.subheader("📊 Validation Metrics")
            trained_model = YOLO(model_path)
            metrics = trained_model.val()
            st.write(f"**mAP50-95:** `{metrics.box.map:.4f}`")
            st.write(f"**mAP50:** `{metrics.box.map50:.4f}`")
            with st.expander("Show all validation metrics"):
                st.json(metrics.results_dict)

        # --- Inference ---
        with st.spinner("Running inference on test images..."):
            st.subheader("🖼️ Inference Results")
            test_images_path = os.path.join(DST_DATASET_PATH, "test/images")
            results = trained_model.predict(source=test_images_path, conf=conf_threshold)
            
            for i, result in enumerate(results[:6]): # Show first 6 images
                predicted_image_bgr = result.plot()
                predicted_image_rgb = predicted_image_bgr[..., ::-1] # BGR to RGB
                col1, col2 = st.columns(2)
                col1.image(result.path, caption=f"Original: {os.path.basename(result.path)}", use_column_width=True)
                col2.image(predicted_image_rgb, caption=f"Prediction: {os.path.basename(result.path)}", use_column_width=True)
            st.success("✅ Inference complete!")
