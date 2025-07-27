import torch

path = 'trained_model.pth'  # If you put it in the same folder

try:
    model = torch.load(path)
    print("✅ Model loaded successfully (full model object).")
    print(type(model))
except Exception as e:
    print("❌ Failed to load as full model object.")
    print("Trying as state_dict...")
    try:
        state_dict = torch.load(path)
        print("✅ Loaded as state_dict.")
        print(state_dict.keys())
    except Exception as e2:
        print("❌ Failed to load as state_dict too.")
        print("Error:", e2)