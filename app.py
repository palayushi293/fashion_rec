from flask import Flask, render_template, request
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import os

app = Flask(__name__)


model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")


IMAGE_FOLDER = "static/tshirts"
image_files = [os.path.join(IMAGE_FOLDER, f) 
               for f in os.listdir(IMAGE_FOLDER) 
               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]


images = []
for img_path in image_files:
    try:
        img = Image.open(img_path).convert("RGB")
        images.append(img)
    except Exception as e:
        print(f"Failed to load {img_path}: {e}")

print(f"{len(images)} images loaded successfully")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form["query"]

      
        inputs = processor(text=[query], images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=0)

        scores = probs.squeeze().tolist()
        results = list(zip(image_files, scores))
        
    
        positive_results = [(img, score) for img, score in results if score > 0]
        positive_results.sort(key=lambda x: x[1], reverse=True)

        return render_template("results.html", query=query, results=positive_results)

   
    return render_template("index.html", all_images=image_files)

if __name__ == "__main__":
    app.run(debug=True)
