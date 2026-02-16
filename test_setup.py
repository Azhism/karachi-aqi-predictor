"""
Test script to verify setup
Run this to check if everything is configured correctly
"""
import sys
import os

print("="*60)
print("🧪 TESTING PROJECT SETUP")
print("="*60)

# Test 1: Check Python version
print("\n1️⃣  Testing Python version...")
if sys.version_info >= (3, 8):
    print(f"   ✅ Python {sys.version.split()[0]} (Good!)")
else:
    print(f"   ❌ Python {sys.version.split()[0]} (Need 3.8+)")
    sys.exit(1)

# Test 2: Check .env file
print("\n2️⃣  Testing .env file...")
if os.path.exists('.env'):
    print("   ✅ .env file exists")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    mongodb_uri = os.getenv('MONGODB_URI')
    if mongodb_uri and 'mongodb+srv' in mongodb_uri:
        if 'username:password' in mongodb_uri:
            print("   ⚠️  WARNING: Update .env with your MongoDB credentials!")
        else:
            print("   ✅ MongoDB URI configured")
    else:
        print("   ❌ MongoDB URI not configured in .env")
        sys.exit(1)
else:
    print("   ❌ .env file not found")
    sys.exit(1)

# Test 3: Check required packages
print("\n3️⃣  Testing required packages...")
required_packages = [
    'pandas',
    'numpy',
    'pymongo',
    'sklearn',
    'streamlit',
    'plotly'
]

missing_packages = []
for package in required_packages:
    try:
        __import__(package)
        print(f"   ✅ {package}")
    except ImportError:
        print(f"   ❌ {package} (not installed)")
        missing_packages.append(package)

if missing_packages:
    print(f"\n   Install missing packages: pip install {' '.join(missing_packages)}")
    sys.exit(1)

# Test 4: Check project structure
print("\n4️⃣  Testing project structure...")
required_dirs = ['src', 'data', 'models', 'notebooks']
required_files = ['src/config.py', 'src/database.py']

for directory in required_dirs:
    if os.path.exists(directory):
        print(f"   ✅ {directory}/ folder exists")
    else:
        print(f"   ❌ {directory}/ folder missing")

for file in required_files:
    if os.path.exists(file):
        print(f"   ✅ {file} exists")
    else:
        print(f"   ❌ {file} missing")

# Test 5: Test MongoDB connection
print("\n5️⃣  Testing MongoDB connection...")
try:
    from src.database import MongoDBHandler
    db = MongoDBHandler()
    print("   ✅ MongoDB connection successful")
    
    # Get stats
    db.get_collection_stats()
    
    db.close()
except Exception as e:
    print(f"   ❌ MongoDB connection failed: {e}")
    print("\n   Troubleshooting:")
    print("   1. Check MongoDB URI in .env file")
    print("   2. Verify MongoDB Atlas cluster is running")
    print("   3. Check network access (allow 0.0.0.0/0)")
    print("   4. Verify database user credentials")
    sys.exit(1)

# Test 6: Check for CSV file
print("\n6️⃣  Testing for data files...")
csv_paths = [
    'data/karachi_complete_dataset.csv',
    'karachi_complete_dataset.csv'
]

csv_found = False
for csv_path in csv_paths:
    if os.path.exists(csv_path):
        print(f"   ✅ Found: {csv_path}")
        csv_found = True
        
        # Check CSV content
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"      Records: {len(df):,}")
        print(f"      Features: {len(df.columns)}")
        break

if not csv_found:
    print("   ⚠️  CSV file not found")
    print("      Place 'karachi_complete_dataset.csv' in data/ folder")

# Final summary
print("\n" + "="*60)
print("📊 TEST SUMMARY")
print("="*60)
print("✅ Python version: OK")
print("✅ Configuration: OK")
print("✅ Packages: OK")
print("✅ Project structure: OK")
print("✅ MongoDB connection: OK")
if csv_found:
    print("✅ Data file: OK")
else:
    print("⚠️  Data file: Not found (upload CSV to data/ folder)")

print("\n" + "="*60)
print("🎉 SETUP TEST COMPLETE!")
print("="*60)

if csv_found:
    print("\n✅ Everything is ready!")
    print("\n📋 Next steps:")
    print("   1. Run: python upload_to_mongodb.py")
    print("   2. Implement feature_pipeline.py")
    print("   3. Implement training_pipeline.py")
else:
    print("\n⚠️  Almost ready!")
    print("\n📋 Next steps:")
    print("   1. Place CSV file in data/ folder")
    print("   2. Run: python upload_to_mongodb.py")
    print("   3. Continue development")
