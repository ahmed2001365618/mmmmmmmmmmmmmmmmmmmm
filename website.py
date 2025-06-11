import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile


st.set_page_config(page_title="Multi-Model Detector", layout="centered")
st.title("🧠 Mdical and electrical images detection")
st.write("Select the type of detection you want, then upload an image.")


model_option = st.selectbox(
    "🔍 Select Detection Type:",
    ("pcb smd Detection", "Brain Tumor Detection", "Broken Bone Detection", "eye suger")
)


detection_model = None
classification_model = None

if model_option == "Brain Tumor Detection":
    detection_model = YOLO("brain_tumor_yolo.pt")    
    classification_model = YOLO("best.pt")             
else:
    model_paths = {
        "pcb smd Detection": "smd111.pt",
        "Broken Bone Detection": "bone.pt",
        "eye suger": "suger2.pt"
    }
    detection_model = YOLO(model_paths[model_option])


uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

   
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        image_path = temp.name

    with st.spinner("Running object detection..."):
        detection_results = detection_model(image_path)
        detection_image = detection_results[0].plot()
        boxes = detection_results[0].boxes

    
    classification_label = None
    if classification_model:
        with st.spinner("🧠 Classifying tumor type..."):
            cls_results = classification_model.predict(image_path, task="classify")
            class_id = int(cls_results[0].probs.top1)
            class_conf = float(cls_results[0].probs.top1conf)
            classification_label = f"{classification_model.names[class_id]} ({class_conf:.2%})"

   
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(" Uploaded Image")
        st.image(image, use_column_width=True)
    with col2:
        st.subheader("📍 Detection Result")
        st.image(detection_image, use_column_width=True)

   
    with st.expander(" Detection Details"):
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = detection_model.names[cls_id]
                st.write(f"- **{label}**: {confidence:.2%}")
        else:
            st.write("❌ No objects detected.")

   
    if classification_label:
        st.markdown("### 🧬 Tumor Classification")
        st.success(f"Predicted Type: **{classification_label}**")
