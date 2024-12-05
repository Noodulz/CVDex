import torch
import torch.nn as nn
import torch.nn.functional as F
import kagglehub
from torchvision import transforms, datasets, models
from PIL import Image
import io
from flask import Flask, request, jsonify

app = Flask(__name__)

class PokeModel(nn.Module):
    def __init__(self, num_labels) -> None:
        super(PokeModel, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_labels)
    
    def forward(self, x):
        return self.resnet(x)

width = height = 224
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
    transforms.Resize((width, height)),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

path = kagglehub.dataset_download("lantian773030/pokemonclassification")
dataset = datasets.ImageFolder(root=path+"/PokemonData", transform=transform)
num_labels = len(dataset.classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lmodel = PokeModel(num_labels=num_labels)
lmodel.load_state_dict(torch.load("PokemonModel.pth", weights_only=True))
lmodel.to(device)

print(dataset.classes)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"Error": "No image file provided"}), 400

        img_file = request.files["image"]
        image = Image.open(io.BytesIO(img_file.read())).convert("RGB")

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = lmodel(input_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)

        label = dataset.classes[pred.item()]
        confidence_percent = confidence.item() * 100

        return jsonify({"label": label, "confidence": f"{confidence_percent:.3f}"}), 200

    except Exception as e:
        return jsonify({"Error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)