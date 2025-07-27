import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import sys

def test_single_image(image_path):
    """Test a single image directly"""
    
    # Initialize device and class names
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_names = ['disturbing_content', 'drug_use', 'horror', 'neutral', 'nudity', 'violence']
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load model
    print("Loading model...")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 6)
    state_dict = torch.load('trained_model.pth', map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("✅ Model loaded!")
    
    # Test the image
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        # Get top 3 predictions
        top3_prob, top3_idx = torch.topk(probabilities, 3)
        
        print(f"\n📸 Image: {image_path}")
        print("🎯 Predictions:")
        print("-" * 40)
        for i in range(3):
            class_name = class_names[top3_idx[i]]
            confidence = top3_prob[i].item()
            bar = "█" * int(confidence * 20)
            print(f"   {class_name:18} {confidence:.3f} {bar}")
            
        # Show the top prediction clearly
        top_class = class_names[top3_idx[0]]
        top_confidence = top3_prob[0].item()
        print(f"\n🏆 TOP PREDICTION: {top_class} ({top_confidence:.3f})")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line usage: python quick_test.py "image_path"
        image_path = sys.argv[1]
        test_single_image(image_path)
    else:
        # Interactive usage
        print("🔍 Quick Image Tester")
        print("=" * 30)
        image_path = input("Enter image path: ").strip().strip('"')
        test_single_image(image_path)
