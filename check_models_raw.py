"""Check what's in MongoDB models collection"""
from src.database import MongoDBHandler
import json

db = MongoDBHandler()

print("🔍 MONGODB MODELS COLLECTION RAW DATA\n")

models = list(db.models.find())

if not models:
    print("❌ No models found")
else:
    print(f"✅ Found {len(models)} model documents\n")
    
    for i, model in enumerate(models, 1):
        print(f"\n{'='*80}")
        print(f"Model {i}:")
        print('='*80)
        
        # Print all fields
        for key, value in model.items():
            if key == '_id':
                print(f"  _id: {value}")
            else:
                print(f"  {key}: {value}")

db.close()
