import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from deepface import DeepFace
import tempfile

# App Configuration
st.set_page_config(page_title="Multi-Model Detector", layout="centered")
st.title("🧠 Medical and Electrical ")
st.write("Select the type of detection you want, then upload an image.")

# Model configuration
MODEL_CONFIG = {
    "pcb smd Detection": {"path": "smd111.pt", "threshold": 0.75, "color": "red"},
    "Brain Tumor Detection": {
        "detection_path": "brain_tumor_yolo.pt",
        "classification_path": "best.pt",
        "detection_threshold": 0.2,
        "classification_threshold": 0.75,
        "color": "red"
    },
    "Broken Bone Detection": {"path": "bone.pt", "threshold": 0.50, "color": "red"},
    "eye suger": {"path": "suger2.pt", "threshold": 0.50, "color": "red"},
    "Teeth Detection": {"path": "teeth.pt", "threshold": 0.20, "color": "red"},
    "Age, Gender & Health Detection": {"task": "deepface"}  
}


model_option = st.selectbox(" Select Detection Type:", list(MODEL_CONFIG.keys()))
config = MODEL_CONFIG[model_option]

# File uploader
uploaded_file = st.file_uploader(" Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Save image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        image_path = temp.name

    if model_option == "Age, Gender & Health Detection":
        with st.spinner("Analyzing face attributes..."):
            try:
                result = DeepFace.analyze(img_path=image_path, actions=["age", "gender", "emotion"], enforce_detection=False)[0]

                # Display results
                st.subheader("Face Attribute Detection")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                with col2:
                    st.success(f"**Estimated Age:** {result['age']}")
                    st.info(f"**Gender:** {result['gender']}")
                    st.warning(f"**Dominant Emotion (Health):** {result['dominant_emotion']}")

            except Exception as e:
                st.error(f"Face detection failed: {e}")

    else:
        # Load detection model(s)
        detection_model = None
        classification_model = None

        if model_option == "Brain Tumor Detection":
            detection_model = YOLO(config["detection_path"])
            classification_model = YOLO(config["classification_path"])
        else:
            detection_model = YOLO(config["path"])

        # Run detection
        with st.spinner(" Running object detection..."):
            detection_results = detection_model(image_path)
            boxes = detection_results[0].boxes

            detection_image = image.copy()
            draw = ImageDraw.Draw(detection_image)

            try:
                font = ImageFont.truetype("arial.ttf", size=20)
            except:
                font = ImageFont.load_default()

            threshold_display = config.get("detection_threshold", config.get("threshold", 0.5))
            box_color = config.get("color", "red")

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf >= threshold_display:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        class_id = int(box.cls[0])
                        label = f"{detection_model.names[class_id]} ≥ {threshold_display:.0%}"
                        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
                        draw.text((x1, y1 - 20), label, fill=box_color, font=font)

        # Classification (Brain Tumor)
        classification_label = None
        classification_conf = None
        if classification_model:
            with st.spinner("🧠 Classifying tumor type..."):
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
            st.subheader("Detection Result")
            st.image(detection_image, use_column_width=True)
            st.caption(f"Showing predictions ≥ {threshold_display:.0%} confidence")

        # Detection Details
        with st.expander("Detection Details"):
            if boxes is not None and len(boxes) > 0:
                high_conf_detections = [box for box in boxes if float(box.conf[0]) >= threshold_display]
                if high_conf_detections:
                    st.write(f"**Detected Objects (≥ {threshold_display:.0%} confidence):**")
                    for box in high_conf_detections:
                        cls_id = int(box.cls[0])
                        label = detection_model.names[cls_id]
                        confidence = float(box.conf[0])
                        st.success(f"- {label}: {confidence:.2%} confidence")
                else:
                    st.warning("No objects detected above threshold.")
            else:
                st.warning("No objects detected.")

        # Tumor classification
        if classification_label and classification_conf:
            st.markdown(f"### 🧬 Tumor Classification (≥ {config['classification_threshold']:.0%} confidence)")
            st.success(f"**Predicted Type:** {classification_label} ({classification_conf:.2%})")
