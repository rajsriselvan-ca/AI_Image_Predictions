import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import glob
from collections import defaultdict

def test_model_on_all_images():
    """Test the model on all images in the Predict folder"""
    
    print("🚀 Testing Model on ALL Images")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_names = ['disturbing_content', 'drug_use', 'horror', 'neutral', 'nudity', 'violence']
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load model
    print("📥 Loading model...")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 6)
    state_dict = torch.load('trained_model.pth', map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("✅ Model loaded successfully!")
    
    # Test on all images
    predict_folder = 'Predict'
    if not os.path.exists(predict_folder):
        print("❌ Predict folder not found!")
        return
    
    # Get all categories
    categories = [d for d in os.listdir(predict_folder) 
                 if os.path.isdir(os.path.join(predict_folder, d))]
    
    print(f"\n📁 Found {len(categories)} categories: {', '.join(categories)}")
    
    # Results tracking
    total_correct = 0
    total_images = 0
    category_results = {}
    all_predictions = []
    
    # Test each category
    for category in categories:
        print(f"\n{'='*60}")
        print(f"📂 Testing Category: {category.upper()}")
        print('='*60)
        
        category_path = os.path.join(predict_folder, category)
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(category_path, ext)))
        
        if not image_files:
            print(f"   ⚠️  No images found in {category}")
            continue
        
        print(f"   🖼️  Found {len(image_files)} images")
        
        correct_predictions = 0
        category_predictions = []
        
        # Test each image
        for i, image_file in enumerate(image_files, 1):
            try:
                # Load and preprocess image
                image = Image.open(image_file).convert('RGB')
                input_tensor = transform(image).unsqueeze(0).to(device)
                
                # Make prediction
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    confidence, predicted = torch.max(probabilities, 0)
                
                predicted_class = class_names[predicted.item()]
                confidence_score = confidence.item()
                
                # Check if correct
                is_correct = predicted_class == category
                if is_correct:
                    correct_predictions += 1
                    total_correct += 1
                
                total_images += 1
                
                # Store prediction
                prediction_info = {
                    'image': os.path.basename(image_file),
                    'true_class': category,
                    'predicted_class': predicted_class,
                    'confidence': confidence_score,
                    'correct': is_correct
                }
                category_predictions.append(prediction_info)
                all_predictions.append(prediction_info)
                
                # Print result
                status = "✅" if is_correct else "❌"
                print(f"   {status} {i:3d}. {os.path.basename(image_file):30} -> {predicted_class:18} ({confidence_score:.3f})")
                
            except Exception as e:
                print(f"   ❌ Error processing {os.path.basename(image_file)}: {e}")
        
        # Calculate category accuracy
        category_accuracy = (correct_predictions / len(image_files) * 100) if image_files else 0
        category_results[category] = {
            'correct': correct_predictions,
            'total': len(image_files),
            'accuracy': category_accuracy,
            'predictions': category_predictions
        }
        
        print(f"\n   📊 Category Summary:")
        print(f"   Accuracy: {category_accuracy:.1f}% ({correct_predictions}/{len(image_files)})")
    
    # Overall Results
    print(f"\n{'='*80}")
    print("📈 OVERALL RESULTS")
    print('='*80)
    
    if total_images > 0:
        overall_accuracy = total_correct / total_images * 100
        print(f"🏆 Overall Accuracy: {overall_accuracy:.1f}% ({total_correct}/{total_images})")
        
        print(f"\n📋 Category Breakdown:")
        print("-" * 60)
        print(f"{'Category':<20} {'Accuracy':<12} {'Correct/Total':<15}")
        print("-" * 60)
        
        for category, results in category_results.items():
            accuracy = results['accuracy']
            correct = results['correct']
            total = results['total']
            print(f"{category:<20} {accuracy:>7.1f}%     {correct:>3}/{total:<3}")
        
        # Confusion Matrix Summary
        print(f"\n🔄 Common Misclassifications:")
        print("-" * 50)
        
        misclassifications = defaultdict(int)
        for pred in all_predictions:
            if not pred['correct']:
                key = f"{pred['true_class']} -> {pred['predicted_class']}"
                misclassifications[key] += 1
        
        for error, count in sorted(misclassifications.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {error:<35} ({count} times)")
        
        # Best and Worst Performing Images
        print(f"\n⭐ Best Predictions (Highest Confidence):")
        correct_preds = [p for p in all_predictions if p['correct']]
        if correct_preds:
            best_preds = sorted(correct_preds, key=lambda x: x['confidence'], reverse=True)[:3]
            for pred in best_preds:
                print(f"   ✅ {pred['image']:25} -> {pred['predicted_class']:15} ({pred['confidence']:.3f})")
        
        print(f"\n⚠️  Worst Predictions (Confident but Wrong):")
        wrong_preds = [p for p in all_predictions if not p['correct']]
        if wrong_preds:
            worst_preds = sorted(wrong_preds, key=lambda x: x['confidence'], reverse=True)[:3]
            for pred in worst_preds:
                print(f"   ❌ {pred['image']:25} -> {pred['predicted_class']:15} ({pred['confidence']:.3f}) [True: {pred['true_class']}]")
    
    else:
        print("❌ No images were processed!")
    
    print(f"\n{'='*80}")
    print("✅ Testing Complete!")
    print('='*80)

if __name__ == "__main__":
    test_model_on_all_images()
