import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
from torchvision import transforms
from GlaucomaModel import UNet, vertical_cup_to_disc_ratio, refine_seg
import io

app = Flask(__name__)
CORS(app)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=3, n_classes=2).to(device)
checkpoint = torch.load('unet_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Threshold for vCDR classification
vCDR_threshold = 0.6

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Valid file extensions
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

def process_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0).to(device)
    return img

@app.route('/api/calculate_vcdr', methods=['POST'])
def calculate_vcdr():
    if 'images' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist('images')
    if not files:
        return jsonify({"error": "No files received"}), 400

    results = []
    for file in files:
        # Check file extension
        if not file.filename.lower().endswith(valid_extensions):
            results.append({"file": file.filename, "error": "Unsupported file type"})
            continue

        image_bytes = file.read()
        img_tensor = process_image(image_bytes)

        with torch.no_grad():
            logits = model(img_tensor)

        # Get segmentation predictions for OD and OC
        pred_od = (logits[:, 0, :, :] >= 0.5).type(torch.int8)
        pred_oc = (logits[:, 1, :, :] >= 0.5).type(torch.int8)

        # Refine segmentations
        pred_od_refined = refine_seg(pred_od.cpu()).to(device)
        pred_oc_refined = refine_seg(pred_oc.cpu()).to(device)

        # Compute vCDR
        pred_vCDR = vertical_cup_to_disc_ratio(pred_od_refined.cpu().numpy(), pred_oc_refined.cpu().numpy())[0]

        # Classify based on vCDR threshold
        predicted_label = "Glaucoma" if pred_vCDR > vCDR_threshold else "No Glaucoma"

        # Append results with vCDR and prediction
        results.append({
            "file": file.filename,
            "vCDR": f"{pred_vCDR:.2f}",
            "prediction": predicted_label
        })

    return jsonify({"results": results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
