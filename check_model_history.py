"""Check model training history in MongoDB"""
from src.database import MongoDBHandler
from datetime import datetime

db = MongoDBHandler()

print("📊 MODEL TRAINING HISTORY\n")

# Get all model records from MongoDB
models = list(db.models.find().sort("trained_at", -1))

if not models:
    print("❌ No models found in database")
else:
    print(f"✅ Found {len(models)} model records\n")
    print("=" * 80)
    
    for i, model in enumerate(models, 1):
        print(f"\n{i}. {model.get('model_name', 'Unknown')}")
        print(f"   Trained: {model.get('created_at', 'N/A')}")
        print(f"   Accuracy: {model.get('metrics', {}).get('accuracy', 0):.3f}")
        print(f"   F1 Score: {model.get('metrics', {}).get('f1_score', 0):.3f}")
        print(f"   CV Accuracy: {model.get('metrics', {}).get('cv_accuracy', 0):.3f}")
        
        # Show if it's the best model
        if model.get('is_best'):
            print(f"   🥇 BEST MODEL")
        
        # Show file path
        print(f"   File: {model.get('model_path', 'N/A')}")

    # Show latest training date
    print("\n" + "=" * 80)
    latest = models[0]
    print(f"\n📅 Last Training: {latest.get('created_at', 'Unknown')}")
    
    # Find best model
    best_model = next((m for m in models if m.get('is_best')), models[0])
    print(f"🥇 Best Model: {best_model.get('model_name', 'Unknown')} (Accuracy: {best_model.get('metrics', {}).get('accuracy', 0):.3f})")
    print(f"📁 Model File: {best_model.get('model_path', 'N/A')}")

db.close()
