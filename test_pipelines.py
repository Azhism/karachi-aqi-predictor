"""
Test the feature and training pipelines
"""

print("="*60)
print("🧪 TESTING PIPELINES")
print("="*60)

# Test 1: Feature Pipeline
print("\n1️⃣  Testing Feature Pipeline...")
from src.feature_pipeline import FeaturePipeline

fp = FeaturePipeline()
success = fp.run_hourly_update()
fp.close()

if success:
    print("   ✅ Feature pipeline works!")
else:
    print("   ❌ Feature pipeline failed!")

# Test 2: Training Pipeline
print("\n2️⃣  Testing Training Pipeline...")
from src.training_pipeline import TrainingPipeline

tp = TrainingPipeline()
success = tp.run()
tp.close()

if success:
    print("   ✅ Training pipeline works!")
else:
    print("   ❌ Training pipeline failed!")

print("\n" + "="*60)
print("🎉 PIPELINE TESTING COMPLETE!")
print("="*60)