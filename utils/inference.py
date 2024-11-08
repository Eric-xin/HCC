import torch
from PIL import Image
import torchvision.transforms as transforms

from src.CNN_model import CNNModel

torch.backends.quantized.engine = 'qnnpack'

def Inference(image_paths, model, model_path, device='cpu', quantized=False):
    # Load the model
    if quantized:
        model = torch.quantization.quantize_dynamic(
            torch.load(model_path, map_location=device), 
            {torch.nn.Linear}, 
            dtype=torch.qint8
        )
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor()
    ])
    
    predictions = []
    
    for image_path in image_paths:
        # Load and preprocess the image
        image = Image.open(image_path)
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)  # Convert numpy array to PIL Image
        image = Image.open(image_path).convert('L')
        image = transform(image).unsqueeze(0)  # Transform and add batch dimension
        image = image.to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(image)
        
        # Convert output to numpy array and store the prediction
        output = output.cpu().numpy().flatten()
        predictions.append(output)
    
    return predictions