import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import tempfile

# App Configuration
st.set_page_config(page_title="Multi-Model Detector", layout="centered")
st.title(" Medical Images Detection")
st.write("Select the type of detection you want, then upload an image.")

# Model configuration with custom confidence thresholds
MODEL_CONFIG = {
    "pcb smd Detection": {"path": "smd111.pt", "threshold": 0.4, "color": "red"},
    "Brain Tumor Detection": {
        "detection_path": "brain_tumor_yolo.pt",
        "classification_path": "best.pt",
        "detection_threshold": 0.2,
        "classification_threshold": 0.4,
        "color": "red"
    },
    "Broken Bone Detection": {"path": "bone.pt", "threshold": 0.50, "color": "red"},
    "eye suger": {"path": "suger2.pt", "threshold": 0.50, "color": "red"},
    "Teeth Detection": {"path": "teeth.pt", "threshold": 0.20, "color": "red"}
}

# Model selection
model_option = st.selectbox(" Select Detection Type:", list(MODEL_CONFIG.keys()))

# Load models
config = MODEL_CONFIG[model_option]
detection_model = None
classification_model = None

if model_option == "Brain Tumor Detection":
    detection_model = YOLO(config["detection_path"])
    classification_model = YOLO(config["classification_path"])
else:
    detection_model = YOLO(config["path"])

# File uploader
uploaded_file = st.file_uploader(" Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Save image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        image_path = temp.name

    # Run detection
    with st.spinner("Running object detection..."):
        detection_results = detection_model(image_path)
        boxes = detection_results[0].boxes

        # Create a clean copy of the original image for custom drawing
        detection_image = image.copy()
        draw = ImageDraw.Draw(detection_image)

        # Try to load a readable font
        try:
            font = ImageFont.truetype("arial.ttf", size=20)
        except:
            font = ImageFont.load_default()

        # Set threshold
        threshold_display = config.get("detection_threshold", config.get("threshold", 0.5))
        box_color = config.get("color", "red")

        # Draw filtered boxes manually
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                conf = float(box.conf[0])
                if conf >= threshold_display:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    label = f"{detection_model.names[class_id]} ≥ {threshold_display:.0%}"
                    draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
                    draw.text((x1, y1 - 20), label, fill=box_color, font=font)

    # Classification (only for Brain Tumor)
    classification_label = None
    classification_conf = None
    if classification_model:
        with st.spinner(" Classifying tumor type..."):
            cls_results = classification_model.predict(image_path, task="classify")
            class_id = int(cls_results[0].probs.top1)
            class_conf = float(cls_results[0].probs.top1conf)
            if class_conf >= config["classification_threshold"]:
                classification_label = classification_model.names[class_id]
                classification_conf = class_conf

    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(" Uploaded Image")
        st.image(image, use_column_width=True)
    with col2:
        st.subheader(" Detection Result")
        st.image(detection_image, use_column_width=True)
        
        # Display model-specific confidence threshold
        threshold = config.get("detection_threshold", config.get("threshold"))
        st.caption(f"Showing predictions ≥ {threshold:.0%} confidence")

    # Detection details
    with st.expander(" Detection Details"):
        if boxes is not None and len(boxes) > 0:
            high_conf_detections = [box for box in boxes if float(box.conf[0]) >= threshold]
            
            if high_conf_detections:
                st.write(f"**Detected Objects (≥ {threshold:.0%} confidence):**")
                for box in high_conf_detections:
                    cls_id = int(box.cls[0])
                    label = detection_model.names[cls_id]
                    confidence = float(box.conf[0])
                    st.success(f"- {label}: {confidence:.2%} confidence")
            else:
                st.warning(f"No objects detected with ≥ {threshold:.0%} confidence")
        else:
            st.warning("No objects detected")

    # Tumor classification result (if available)
    if classification_label and classification_conf:
        st.markdown(f"### 🧬 Tumor Classification (≥ {config['classification_threshold']:.0%} confidence)")
        st.success(f"**Predicted Type:** {classification_label} ({classification_conf:.2%})")
