from src.database import MongoDBHandler
import pandas as pd

db = MongoDBHandler()

print("="*70)
print("📊 MODELS IN MONGODB")
print("="*70)

models = list(db.models.find().sort('timestamp', -1))
print(f"\nTotal models: {len(models)}")

if len(models) > 0:
    print("\n📋 Latest models:")
    
    for model in models:
        name = model.get('model_name', 'Unknown')
        best = '🥇' if model.get('is_best', False) else '  '
        metrics = model.get('metrics', {})
        
        print(f"{best} {name:15s} | Acc: {metrics.get('test_accuracy', 0):.3f} | F1: {metrics.get('test_f1', 0):.3f} | Created: {model.get('created_at', 'N/A')}")
    
    best = [m for m in models if m.get('is_best', False)]
    if best:
        best_metrics = best[0].get('metrics', {})
        print(f"\n🥇 Best model: {best[0].get('model_name', 'Unknown')}")
        print(f"   Test Accuracy: {best_metrics.get('test_accuracy', 0):.3f}")
        print(f"   Test F1 Score: {best_metrics.get('test_f1', 0):.3f}")
        print(f"   CV Accuracy: {best_metrics.get('cv_accuracy', 0):.3f}")
else:
    print("❌ No models found in MongoDB")

db.close()
