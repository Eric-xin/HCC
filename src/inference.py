import torch
from PIL import Image
# import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def Inference(image_paths, model, model_path, device='cpu'):
    # Load the model
    model.load_state_dict(torch.load(model_path, map_location=device))
    # model.load_state_dict(torch.load('models/CNN_100_epoch_model.pth', map_location=device))
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

# # Example usage
# from CNN_model import CNNModel

# test_img = 'data/hydrocarbon/images/146002807.png'
# model = CNNModel()
# test_prediction = Inference(image_paths=[test_img], model=model, model_path='models/CNN_100_epoch_model.pth')
# print(test_prediction)