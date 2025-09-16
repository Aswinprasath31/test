import streamlit as st
import os
import shutil
import yaml
from ultralytics import YOLO
import glob
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="🚗 YOLOv8 Car Detection",
    page_icon="🚗",
    layout="wide"
)

# --- PATHS & CONFIG ---
# NOTE: This app assumes your dataset is in the 'car_dataset-master' directory
# in the same folder as this 'app.py' script.
SRC_DATASET_PATH = "car_dataset-master"
WORKING_DIR = "working"
DST_DATASET_PATH = os.path.join(WORKING_DIR, "datasets/car-detection-dataset")
YAML_SAVE_PATH = os.path.join(WORKING_DIR, "data.yaml")
PROJECT_NAME = "car_detection_yolov8"

# Ensure the working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)


# --- HELPER FUNCTIONS ---
def setup_dataset():
    """Copies dataset to a writable directory and creates the YAML file."""
    if not os.path.exists(SRC_DATASET_PATH):
        st.error(f"Source dataset not found at: '{SRC_DATASET_PATH}'. Please ensure the dataset is in the correct location.")
        return False

    with st.spinner("Preparing dataset... This may take a moment."):
        # Remove previous working data if it exists
        if os.path.exists(DST_DATASET_PATH):
            shutil.rmtree(DST_DATASET_PATH)

        # Copy dataset to the working directory
        shutil.copytree(SRC_DATASET_PATH, DST_DATASET_PATH)

        # Create the data.yaml file for YOLO
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
    st.success(f"✅ Dataset is ready and data.yaml is created at {YAML_SAVE_PATH}.")
    return True

def find_latest_run_dir(project_name):
    """Finds the latest run directory within the project."""
    list_of_dirs = glob.glob(os.path.join('runs/detect', project_name + '*'))
    if not list_of_dirs:
        return None
    return max(list_of_dirs, key=os.path.getmtime)


# --- STREAMLIT APP LAYOUT ---

st.title("🚗 YOLOv8 Car Detection Trainer & Inspector")

st.markdown("""
This application walks you through training a YOLOv8 model on a custom car dataset.
Follow the steps below to prepare the data, train the model, and view the predictions.
""")

# --- Initialize Session State ---
if 'dataset_ready' not in st.session_state:
    st.session_state['dataset_ready'] = False
if 'training_complete' not in st.session_state:
    st.session_state['training_complete'] = False
if 'trained_model' not in st.session_state:
    st.session_state['trained_model'] = None


# --- SIDEBAR FOR CONFIGURATION ---
st.sidebar.header("⚙️ Model & Training Configuration")

selected_model = st.sidebar.selectbox(
    "Choose a YOLOv8 Model",
    ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt'],
    help="'n' is nano (fastest), 's' is small, 'm' is medium (more accurate)."
)

epochs = st.sidebar.slider("Number of Epochs", 1, 100, 25)
img_size = st.sidebar.selectbox("Image Size (pixels)", [320, 640], index=1)
batch_size = st.sidebar.slider("Batch Size", 1, 32, 8)
conf_threshold = st.sidebar.slider("Confidence Threshold for Prediction", 0.0, 1.0, 0.25, 0.05)


# --- MAIN WORKFLOW ---

# --- Step 1: Dataset Preparation ---
st.header("Step 1: Prepare Your Dataset")
st.markdown("This step copies your dataset to a writable directory and creates the necessary data.yaml configuration file for YOLOv8.")

if st.button("Prepare Dataset", key="prepare_data"):
    setup_dataset()

if st.session_state['dataset_ready']:
    st.success("Dataset is ready for training.")
    with st.expander("View data.yaml content"):
        with open(YAML_SAVE_PATH, 'r') as f:
            st.code(f.read(), language='yaml')


# --- Step 2: Model Training ---
st.header("Step 2: Train the YOLOv8 Model")
st.markdown("Select your training parameters in the sidebar and click 'Start Training'.")

if st.button("Start Training", key="start_training", disabled=not st.session_state['dataset_ready']):
    with st.spinner(f"Training YOLOv8 model for {epochs} epochs... Please wait. This can take a long time."):
        try:
            # Load the selected YOLO model
            model = YOLO(selected_model)

            # Train the model
            model.train(
                data=YAML_SAVE_PATH,
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                project='runs/detect', # Base project directory
                name=PROJECT_NAME,     # Subdirectory for this run
                exist_ok=True          # Overwrite previous runs with the same name
            )

            # Find the path to the best trained weights
            run_dir = find_latest_run_dir(PROJECT_NAME)
            best_weights_path = os.path.join(run_dir, "weights/best.pt")

            if os.path.exists(best_weights_path):
                st.session_state['training_complete'] = True
                st.session_state['trained_model_path'] = best_weights_path
                st.success(f"✅ Training complete! Model saved at: {best_weights_path}")
            else:
                st.error("Training finished, but the 'best.pt' model file could not be found.")

        except Exception as e:
            st.error(f"An error occurred during training: {e}")


# --- Step 3: Validation & Prediction ---
st.header("Step 3: Evaluate and Predict")
st.markdown("Once training is complete, you can validate the model's performance and run predictions on test images.")

if st.button("Run Validation & Inference", key="run_inference", disabled=not st.session_state.get('training_complete', False)):
    model_path = st.session_state.get('trained_model_path')
    if not model_path or not os.path.exists(model_path):
        st.error("Trained model not found. Please complete training first.")
    else:
        with st.spinner("Loading trained model and running validation..."):
            trained_model = YOLO(model_path)
            
            # --- Validation ---
            st.subheader("📊 Validation Metrics")
            try:
                metrics = trained_model.val()
                # Extract and display key metrics
                st.write(f"**mAP50-95:** {metrics.box.map:.4f}")
                st.write(f"**mAP50:** {metrics.box.map50:.4f}")
                st.write(f"**mAP75:** {metrics.box.map75:.4f}")
                with st.expander("Show all validation metrics"):
                    st.json(metrics.results_dict)
            except Exception as e:
                st.error(f"An error occurred during validation: {e}")


        with st.spinner("Running inference on test images..."):
            # --- Inference ---
            st.subheader("🖼️ Inference Results")
            test_images_path = os.path.join(DST_DATASET_PATH, "test/images")
            
            if not os.path.exists(test_images_path):
                 st.error(f"Test images not found at path: {test_images_path}")
            else:
                try:
                    results = trained_model.predict(
                        source=test_images_path,
                        conf=conf_threshold,
                        save=False # We will display results manually
                    )

                    # Display a few example predictions
                    num_images_to_show = 6
                    image_files = glob.glob(os.path.join(test_images_path, '*.jpg'))[:num_images_to_show]

                    if not results:
                        st.warning("Inference ran, but no objects were detected in the test images.")
                    else:
                        for i, result in enumerate(results[:num_images_to_show]):
                             # .plot() returns a BGR numpy array with detections
                            predicted_image_bgr = result.plot()
                            # Convert BGR to RGB for display in Streamlit
                            predicted_image_rgb = predicted_image_bgr[..., ::-1]

                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(result.path, caption=f"Original: {os.path.basename(result.path)}", use_column_width=True)
                            with col2:
                                st.image(predicted_image_rgb, caption=f"Prediction: {os.path.basename(result.path)}", use_column_width=True)

                    st.success("✅ Inference complete!")

                except Exception as e:
                    st.error(f"An error occurred during inference: {e}")
