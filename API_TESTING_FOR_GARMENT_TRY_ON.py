from clarifai.client.model import Model
from flask import Flask, request, jsonify, send_file,send_from_directory
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import cv2
import matplotlib.pyplot as plt
import requests
from gradio_client import Client, handle_file, file
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Function to refine the mask (thresholding or smoothing)
def refine_mask(mask, threshold=0.5):
    return np.where(mask >= threshold, 1, 0).astype(np.uint8)
# Function to run SAM on an ROI
def run_sam_on_roi(roi):
    # Model constants
    SAM_MODEL_TYPE = "vit_h"  
    SAM_CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
    print("Loading SAM model...")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device="cpu")
    predictor = SamPredictor(sam)
    print("SAM model loaded successfully.")
    input_image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    predictor.set_image(input_image)
    height, width, _ = input_image.shape
    input_point = np.array([[width // 2, height // 2]])  # Center of the ROI
    input_label = np.array([1])  # Positive point for SAM

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )
    sam_mask = masks[0]  # Take the first mask (binary)
    return sam_mask
# Function to run GrabCut on an ROI
def run_grabcut_on_roi(roi):
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    grabcut_rect = (10, 10, roi.shape[1] - 20, roi.shape[0] - 20)

    cv2.grabCut(roi, mask, grabcut_rect, bg_model, fg_model, iterCount=10, mode=cv2.GC_INIT_WITH_RECT)
    refined_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")  # Refine mask
    return refined_mask

# Function to ensemble SAM and GrabCut masks
def ensemble_sam_grabcut(sam_mask, grabcut_mask, alpha=0.5):
    sam_mask = np.array(sam_mask, dtype=np.float32)
    grabcut_mask = np.array(grabcut_mask, dtype=np.float32)

    # Weighted average of the two masks
    combined_mask = alpha * sam_mask + (1 - alpha) * grabcut_mask

    # Refine combined mask (e.g., by thresholding)
    combined_mask = refine_mask(combined_mask, threshold=0.5)

    return combined_mask

def remove_skin_from_mask(masked_image, mask):
    hsv_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)
    skin_mask_inverted = cv2.bitwise_not(skin_mask)
    refined_mask = cv2.bitwise_and(mask, mask, mask=skin_mask_inverted)

    return refined_mask

def extract_and_display_transparent_mask(original_image, ensembled_mask):
    binary_mask = (ensembled_mask > 0).astype(np.uint8)
    rgba_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)
    rgba_image[:, :, 3] = binary_mask * 255
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(rgba_image, cv2.COLOR_BGRA2RGBA))
    plt.title("Extracted ROI of Dress")
    plt.axis("off")
    plt.show()

    return rgba_image

def create_and_store_roi_mask(roi_image):
    # Step 1: Run SAM and GrabCut on the ROI
    sam_mask = run_sam_on_roi(roi_image) 
    grabcut_mask = run_grabcut_on_roi(roi_image) 

    # Step 2: Combine masks using an ensemble method
    combined_mask = ensemble_sam_grabcut(sam_mask, grabcut_mask, alpha=0.5)

    # Step 3: Remove skin regions from the combined mask
    refined_mask = remove_skin_from_mask(roi_image, combined_mask)

    return refined_mask

def extract_and_display_transparent_mask(original_image, ensembled_mask):
    # Ensure the mask is binary (0 and 1 only)
    binary_mask = (ensembled_mask > 0).astype(np.uint8)

    # Convert the original image to RGBA (adding an alpha channel)
    rgba_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)

    # Set the alpha channel to match the mask
    rgba_image[:, :, 3] = binary_mask * 255

    '''# Display the transparent result
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(rgba_image, cv2.COLOR_BGRA2RGBA))
    plt.title("Extracted Masked Area with Transparency")
    plt.axis("off")
    plt.show()'''

    return rgba_image

def is_exactly_one_human(image_path, confidence_threshold=0.70):
    # Model paths
    model_path = "frozen_inference_graph.pb"
    config_path = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

    # Load model
    net = cv2.dnn_DetectionModel(model_path, config_path)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Perform detection
    class_ids, confidences, _ = net.detect(image, confThreshold=0.4, nmsThreshold=0.4)

    # Count humans with confidence above the threshold
    human_count = sum(1 for class_id, confidence in zip(class_ids.flatten(), confidences.flatten())
                      if class_id == 1 and confidence >= confidence_threshold)

    # Return True if exactly one human is detected
    return human_count == 1

app = Flask(__name__)
@app.route('/upload', methods=['POST'])
def process_clothing_image():
    UPLOAD_FOLDER = "input_img_storage"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    cloth_image = request.files['cloth_image']
    if 'cloth_image' not in request.files:
        return jsonify({"error": "Missing cloth_image"}), 400
    file_name = cloth_image.filename
    image_path = os.path.join(UPLOAD_FOLDER, file_name)
    cloth_image.save(image_path)

    # Valid options
    gender_list = ['Male', 'Female']
    size_list = ['S', 'M', 'L']
    category_list = ['Upper-Body', 'Lower-Body', 'Dress']

    # Retrieve gender, size, and dress category from form data
    gender = request.form.get('gender')
    size = request.form.get('size')
    category = request.form.get('category')

    # Check and validate gender
    if gender:
        if gender in gender_list:
            gender = gender.strip().capitalize()
        else:
            return jsonify({"error": "Invalid gender. Valid options: Male, Female"}), 400
    else:
        return jsonify({"error": "Missing gender field"}), 400

    # Check and validate size
    if size:
        if size in size_list:
            size = size.strip().upper()  # Size is typically in uppercase
        else:
            return jsonify({"error": "Invalid size. Valid options: S, M, L"}), 400
    else:
        return jsonify({"error": "Missing size field"}), 400

    # Check and validate category
    if category:
        if category in category_list:
            category = category.strip().capitalize()
        else:
            return jsonify({"error": "Invalid dress_category. Valid options: Lower-Body, Upper-Body, Dress"}), 400
    else:
        return jsonify({"error": "Missing dress_category field"}), 400


    model_url = "https://clarifai.com/clarifai/main/models/apparel-detection"
    detector_model = Model(
        url=model_url,
        pat="1fc718434f51433691094fcc989117ea",
    )
    CONFIDENCE_THRESHOLD = 0.6
    all_detected_classes = []

    '''# Input and store the image
    image_path = input("Enter the full path of the image to test: ")
    if not os.path.exists(image_path):
        print("The specified image does not exist! Please check the path.")
        exit()  # Exit the program if the image does not exist
    else:
        with open(image_path, "rb") as img_file:
            Garment_img = cv2.imread(image_path)
            original_image = Garment_img.copy()
    plt.imshow(cv2.cvtColor(Garment_img, cv2.COLOR_BGR2RGB))  
    plt.axis('off')  # Hide axes
    plt.show()

    # Model constants
    SAM_MODEL_TYPE = "vit_h"  
    SAM_CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
    print("Loading SAM model...")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device="cpu")
    predictor = SamPredictor(sam)
    print("SAM model loaded successfully.")'''

    if gender == "Female":
        if size == "S":
            avatar_image_path = "female_s.png"
        elif size == "M":
            avatar_image_path = "female_m.jpg"
        elif size == "L":
            avatar_image_path = "female_l.jpg"
    elif gender == "Male":
        if size == "S":
            avatar_image_path = "male_s.png"
        elif size == "M":
            avatar_image_path = "male_m.png"
        elif size == "L":
            avatar_image_path = "male_l.jpg"

    if not is_exactly_one_human(image_path, confidence_threshold=0.70) and category == "Dress":
        if gender == "Female":
            if size == "S":
                avatar_image_path = "trail2.jpg"
            elif size == "M":
                avatar_image_path = "trail3.jpg"
            elif size == "L":
                avatar_image_path = "trail1.jpg"

    if not is_exactly_one_human(image_path, confidence_threshold=0.70) and category == "Upper-body" or category == "Lower-body":
        if gender == "Female":
            if size == "S":
                avatar_image_path = "trail4.png"
            elif size == "M":
                avatar_image_path = "trail6.jpg"
            elif size == "L":
                avatar_image_path = "trail5.jpg"
    if is_exactly_one_human(image_path, confidence_threshold=0.70) and category == "Dress":
        url = "https://try-on-diffusion.p.rapidapi.com/try-on-file"
        headers = {
            "x-rapidapi-host": "try-on-diffusion.p.rapidapi.com",
            "x-rapidapi-key": "7554cb4da9mshc43e4b37fbf9910p1ad935jsn61c457d440c7",
        }

        files = {
            "clothing_image": ("clothing_image.jpg", open(image_path, "rb"), "image/jpeg"),
            "avatar_image": ("avatar_image.jpg", open(avatar_image_path, "rb"), "image/jpeg"),
        }
        data = {
            "clothing_prompt": "",
            "avatar_prompt": "",
            "background_prompt": "A white plain background",
            "seed": 42,
        }

        response = requests.post(url, headers=headers, files=files, data=data)

        if response.status_code == 200:
            result = response.content
            with open("result_image.jpg", "wb") as f:
                f.write(response.content)
            print("Image generated successfully and saved as result_image.jpg.")
            result = cv2.imread("result_image.jpg")
            output_image_path = os.path.join(UPLOAD_FOLDER, "result_image.jpg")
            cv2.imwrite(output_image_path, result)
            #plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            #plt.axis("off")
            #plt.show()
        else:
            return jsonify({"error": response.json(), "status_code": response.status_code}), response.status_code

    if not is_exactly_one_human(image_path, confidence_threshold=0.70):
        if os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                prediction_response = detector_model.predict_by_bytes(img_file.read(), input_type="image")

            regions = prediction_response.outputs[0].data.regions
            if not regions:
                print("No clothing items detected in the image.")
            else:
                image = cv2.imread(image_path)
                original_image = image.copy()

                for region in regions:
                    top_row = region.region_info.bounding_box.top_row
                    left_col = region.region_info.bounding_box.left_col
                    bottom_row = region.region_info.bounding_box.bottom_row
                    right_col = region.region_info.bounding_box.right_col

                    height, width, _ = image.shape

                    # Bounding box coordinates with bounds checking
                    x1 = max(0, min(width, int(left_col * width)))
                    y1 = max(0, min(height, int(top_row * height)))
                    x2 = max(0, min(width, int(right_col * width)))
                    y2 = max(0, min(height, int(bottom_row * height)))

                    # Validate the bounding box
                    if x1 >= x2 or y1 >= y2:
                        print(f"Skipping invalid bounding box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                        continue

                    for concept in region.data.concepts:
                        confidence = concept.value
                        if confidence >= CONFIDENCE_THRESHOLD:
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{concept.name}: {confidence:.2f}"
                            cv2.putText(
                                image,
                                label,
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2,
                            )
                            all_detected_classes.append((concept.name, confidence))
                            break

                # Constants
                PADDING_RATIO = 0.02
                PADDING = int(PADDING_RATIO * min(width, height))
                ROI_STORAGE = {}  # Dictionary to store filtered ROIs
                ALLOWED_CLASSES = {"dress", "top", "pants", "skirt"}  # Classes of interest
                CONFIDENCE_THRESHOLD = 0.70  # Assuming this is predefined

                # Variable to store the best ROI
                best_roi = None
                best_clothing_type = None
                best_confidence = 0
                best_coordinates = None
                max_box_area = 0  # To track the largest box

                # Process all detected regions
                for idx, region in enumerate(regions):
                    # Bounding box coordinates
                    top_row = region.region_info.bounding_box.top_row
                    left_col = region.region_info.bounding_box.left_col
                    bottom_row = region.region_info.bounding_box.bottom_row
                    right_col = region.region_info.bounding_box.right_col

                    height, width, _ = original_image.shape

                    # Bounding box coordinates with bounds checking
                    x1 = max(0, min(width, int(left_col * width))) - PADDING
                    y1 = max(0, min(height, int(top_row * height))) - PADDING
                    x2 = max(0, min(width, int(right_col * width))) + PADDING
                    y2 = max(0, min(height, int(bottom_row * height))) + PADDING

                    # Ensure bounding box is valid
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width, x2)
                    y2 = min(height, y2)

                    # Crop the region of interest (ROI)
                    roi = original_image[y1:y2, x1:x2]

                    # Identify clothing type and confidence
                    for concept in region.data.concepts:
                        if concept.value >= CONFIDENCE_THRESHOLD and concept.name in ALLOWED_CLASSES:
                            clothing_type = concept.name
                            confidence = concept.value

                            # Calculate the area of the bounding box (width * height)
                            box_area = (x2 - x1) * (y2 - y1)

                            # Update the largest ROI if the current box is larger
                            if box_area > max_box_area:
                                max_box_area = box_area
                                best_clothing_type = clothing_type
                                best_roi = roi
                                best_confidence = confidence
                                best_coordinates = (x1, y1, x2, y2)

                # After processing all regions, store the best (largest) ROI if found
                if best_clothing_type:
                   # Add the best ROI to ROI_STORAGE
                    ROI_STORAGE = {
                         "best_roi": {
                         "image": best_roi,
                         "coordinates": best_coordinates,
                        "confidence": best_confidence,
                        "class_name": best_clothing_type }}

                    for roi_key, roi_data in ROI_STORAGE.items():
                        roi_image = roi_data["image"]
                        roi_coordinates = roi_data["coordinates"]
                        confidence = roi_data["confidence"]

                        # Run SAM inference on the ROI
                        sam_mask = run_sam_on_roi(roi_image)

                        # Run GrabCut on the ROI
                        grabcut_mask = run_grabcut_on_roi(roi_image)

                        # Ensemble SAM and GrabCut masks
                        combined_mask = ensemble_sam_grabcut(sam_mask, grabcut_mask, alpha=0.5)

                        # Extract and display the transparent masked area
                        cropped_dress_image = extract_and_display_transparent_mask(roi_image, combined_mask)
                        # Save the numpy array as an image
                        cropped_dress_img_path = "cropped_dress_image.png"
                        cv2.imwrite(cropped_dress_img_path, cropped_dress_image)

                        if category == "Lower-body":
                            if clothing_type == "skirt" and gender == "Female":
                                if size == "S":
                                    avatar_image_path = "trail1.jpg"
                                elif size == "M":
                                    avatar_image_path = "model_4.png"
                                elif size == "L":
                                    avatar_image_path = "model_5.jpg"

                            client = Client("https://katiyar48-ootdiffusion-virtualtryonclothing.hf.space/--replicas/kbp2w/")
                            result = client.predict(
                                vton_img=handle_file(avatar_image_path),
                                garm_img=file(cropped_dress_img_path),
                                category="Lower-body",
                                n_samples=1,
                                n_steps=20,
                                image_scale=2,
                                seed=-1,
                                api_name="/process_dc"
                            )
                            print(result)

                        if category == "Upper-body":
                          print("yes")
                          client = Client("Nymbo/Virtual-Try-On")
                          result = client.predict(
                                dict={
                                    "background": file(avatar_image_path),
                                    "layers": [],
                                    "composite": None,
                                },
                                garm_img=file("cropped_dress_image.png"),
                                garment_des="Replace the garment with the given one",
                                is_checked=True,
                                is_checked_crop=False,
                                denoise_steps=30,
                                seed=42,
                                api_name="/tryon"
                            )
                        if category == "Dress":
                            client = Client("yisol/IDM-VTON")
                            try:
                                result = client.predict(
                                    dict={
                                        "background": file(avatar_image_path),  # Human image path
                                        "layers": [],
                                        "composite": None
                                    },
                                    garm_img=file("cropped_dress_image.png"),
                                    garment_des="Replace the garment with the given one",
                                    is_checked=True,
                                    is_checked_crop=False,
                                    denoise_steps=30,
                                    seed=42,
                                    api_name="/tryon"  # API endpoint
                                )

                                # Handle the result
                                print("Try-on Output Image:", result[0])  # Path to try-on output
                                print("Masked Output Image:", result[1])  # Path to masked output

                            except Exception as e:
                                print(f"Error while making the API call: {e}")

    if not is_exactly_one_human(image_path, confidence_threshold=0.70) and (category == "Upper-body" or category == "Dress"):
        print("Output Image Path:", result[0])
        img_path = result[0]
        img = mpimg.imread(img_path)
        output_image_path = os.path.join(UPLOAD_FOLDER, "result_image.jpg")
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(output_image_path, format="jpg", bbox_inches='tight', pad_inches=0)
        plt.close()

    if not is_exactly_one_human(image_path, confidence_threshold=0.70) and (category == "Lower-body"):
        image_path = result['image']
        output_image_path = os.path.join(UPLOAD_FOLDER, "result_image.jpg")
        img = mpimg.imread(image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(output_image_path, format="jpg", bbox_inches='tight', pad_inches=0)
        plt.close()

    return jsonify({
        "message": "Image uploaded and processed successfully!",
        "output_image_path": "result_image.jpg",
        "gender": gender,
        "size": size,
        "dress_category": category
    }), 200



# Endpoint to fetch and display the processed output image
@app.route('/output', methods=['GET'])
def display_output_image():
    UPLOAD_FOLDER = "input_img_storage"
    output_image_path = os.path.join(UPLOAD_FOLDER, "result_image.jpg")

    if not os.path.exists(output_image_path):
        return jsonify({"error": "Output image not found. Please upload an image first."}), 404
    
    return send_file(output_image_path, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
