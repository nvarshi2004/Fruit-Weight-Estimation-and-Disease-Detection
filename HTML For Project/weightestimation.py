import cv2
import numpy as np
import pandas as pd
import joblib
from ultralytics import YOLO
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops

# ===========================
# CONFIG
# ===========================

MODEL_PKL = "final_apple_weight_model.pkl"
YOLO_MODEL = "yolov8x-seg.pt"
APPLE_CLASS_ID = 47
TARGET_SIZE = 640

# ===========================
# LOAD TRAINED MODEL
# ===========================

model_package = joblib.load(MODEL_PKL)
selected_features = model_package["selected_features"]
scaler = model_package["scaler"]
model = model_package["model"]

# Load YOLO
yolo = YOLO(YOLO_MODEL)

# ===========================
# IMAGE PREPROCESS
# ===========================

def resize_keep_ratio(img):
    h, w = img.shape[:2]
    scale = TARGET_SIZE / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))

def segment(img):
    resized = resize_keep_ratio(img)
    r = yolo(resized, verbose=False)[0]

    if r.masks is None:
        return None

    masks = r.masks.data.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy()

    apples = [
        masks[i]
        for i in range(len(masks))
        if int(classes[i]) == APPLE_CLASS_ID
    ]

    if not apples:
        return None

    areas = [np.sum(m > 0.5) for m in apples]
    best = apples[np.argmax(areas)]
    mask = cv2.resize(best, (img.shape[1], img.shape[0]))

    return (mask > 0.5).astype(np.uint8)

# ===========================
# FEATURE FUNCTIONS (Same as Training)
# ===========================

def texture_features(gray, mask):
    pixels = gray[mask == 1]
    if len(pixels) < 100:
        return [0]*5

    pixels = (pixels / 16).astype(np.uint8)
    side = int(np.sqrt(len(pixels)))
    pixels = pixels[:side*side].reshape(side, side)

    glcm = graycomatrix(
        pixels,
        distances=[1],
        angles=[0],
        levels=16,
        symmetric=True,
        normed=True
    )

    return [
        graycoprops(glcm, p)[0,0]
        for p in ["contrast","dissimilarity","homogeneity","energy","correlation"]
    ]

def geometry_features(mask):
    lab = label(mask)
    props = regionprops(lab)
    if not props:
        return [0]*9

    r = max(props, key=lambda x: x.area)
    h, w = mask.shape
    image_area = h*w
    bbox_area = (r.bbox[2]-r.bbox[0])*(r.bbox[3]-r.bbox[1])

    return [
        r.area/image_area,
        r.eccentricity,
        r.extent,
        r.solidity,
        r.perimeter/(h+w),
        r.major_axis_length/w,
        r.minor_axis_length/h,
        r.equivalent_diameter/max(h,w),
        bbox_area/image_area
    ]

# (Keep slice_features and distribution_features same as training)
def slice_features(mask):

    total = np.sum(mask)

    if total == 0:
        return [0] * 20

    h, w = mask.shape
    feats = []

    for i in range(10):
        sl = mask[i * h // 10:(i + 1) * h // 10, :]
        feats.append(np.sum(sl) / total)

    for i in range(10):
        sl = mask[:, i * w // 10:(i + 1) * w // 10]
        feats.append(np.sum(sl) / total)

    return feats
def distribution_features(mask):

    coords = np.column_stack(np.where(mask == 1))

    if len(coords) < 10:
        return [0] * 13

    cx, cy = coords.mean(axis=0)

    d = np.sqrt((coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2)

    return [
        np.mean(d),
        np.std(d),
        np.min(d),
        np.max(d),
        np.percentile(d, 25),
        np.percentile(d, 50),
        np.percentile(d, 75),
        np.var(d),
        np.mean(d ** 2),
        np.mean(np.abs(d - cx)),
        np.mean(np.abs(d - cy)),
        len(coords),
        np.sum(d)
    ]
 # ===========================
 # FEATURE COLUMN DEFINITIONS
 # ===========================

all_columns = [
    "Contrast","Dissimilarity","Homogeneity","Energy","Correlation",
    "Num_Pixels","Eccentricity","Extent","Solidity","Perimeter",
    "Major_Axis","Minor_Axis","Equiv_Diameter","BBox_Area"
] + [f"H_Slice_{i+1}" for i in range(10)] \
  + [f"V_Slice_{i+1}" for i in range(10)] \
  + [f"DF_{i+1}" for i in range(13)]


# ===========================
# PUBLIC API
# ===========================

def predict_weight_from_image_array(img: np.ndarray) -> float:
    """
    Predict apple weight (in grams) from a BGR image array.
    Designed to be called from Flask app with an already-loaded image.
    """
    if img is None:
        raise ValueError("Input image is None.")

    mask = segment(img)
    if mask is None:
        raise ValueError("Apple not detected in the image.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    tex = texture_features(gray, mask)
    geo = geometry_features(mask)
    sli = slice_features(mask)
    dis = distribution_features(mask)

    features = tex + geo + sli + dis

    df = pd.DataFrame([features], columns=all_columns)
    df = df[selected_features]
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)

    return float(prediction[0])
