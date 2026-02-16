# 🚀 SETUP GUIDE FOR KARACHI AQI PREDICTOR

## ✅ What's Been Created

Your project structure is now set up:

```
karachi-aqi-predictor/
├── data/                      # Place your CSV here
├── models/                    # Trained models will be saved here
├── notebooks/                 # Jupyter notebooks
├── src/                       # Source code
│   ├── __init__.py
│   ├── config.py             ✅ Configuration management
│   ├── database.py           ✅ MongoDB operations
│   ├── feature_pipeline.py   ⚠️  To be implemented
│   └── training_pipeline.py  ⚠️  To be implemented
├── .env                      ⚠️  UPDATE WITH YOUR CREDENTIALS
├── .gitignore               ✅ Ready
├── requirements.txt         ✅ Ready
├── upload_to_mongodb.py     ✅ Ready to use
├── app.py                   ⚠️  Dashboard (implement later)
└── README.md                ✅ Ready
```

---

## 📋 STEP-BY-STEP SETUP INSTRUCTIONS

### **Step 1: Setup MongoDB Atlas** 🗄️

1. **Create Account**
   - Go to: https://www.mongodb.com/cloud/atlas
   - Sign up (FREE tier)

2. **Create Cluster**
   - Choose: **M0 Free** tier
   - Region: Choose closest to you
   - Cluster name: `karachi-aqi`

3. **Create Database User**
   - Go to: Database Access → Add New User
   - Username: `karachi_aqi_user`
   - Password: Generate secure password (SAVE IT!)
   - Role: Read and Write to any database

4. **Configure Network Access**
   - Go to: Network Access → Add IP Address
   - Choose: **Allow Access from Anywhere** (0.0.0.0/0)
   - ⚠️ For development only!

5. **Get Connection String**
   - Click: **Connect** → **Connect your application**
   - Copy the connection string:
     ```
     mongodb+srv://karachi_aqi_user:<password>@cluster0.xxxxx.mongodb.net/
     ```
   - Replace `<password>` with your actual password

---

### **Step 2: Update .env File** ⚙️

Open `.env` file and update:

```env
MONGODB_URI=mongodb+srv://karachi_aqi_user:YOUR_PASSWORD@cluster0.xxxxx.mongodb.net/
MONGODB_DATABASE=aqi_karachi

CITY_NAME=Karachi
LATITUDE=24.8608
LONGITUDE=67.0104

PREDICTION_HORIZON=72
TRAIN_TEST_SPLIT=0.2
```

✅ **IMPORTANT**: Replace YOUR_PASSWORD with your actual MongoDB password!

---

### **Step 3: Install Python Dependencies** 📦

Open terminal in VSCode and run:

```powershell
# Create virtual environment
python -m venv venv

# Activate it (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Install dependencies
pip install -r requirements.txt
```

✅ You should see all packages installing

---

### **Step 4: Place Your CSV File** 📂

1. Copy your CSV file from Google Colab
2. Rename it to: `karachi_complete_dataset.csv`
3. Place it in the `data/` folder:
   ```
   data/karachi_complete_dataset.csv
   ```

---

### **Step 5: Test MongoDB Connection** 🧪

In VSCode terminal:

```powershell
python src/database.py
```

✅ Expected output:
```
🔌 Connecting to MongoDB...
✅ Connected to MongoDB database: aqi_karachi
✅ Database indexes created

====================================================
📊 DATABASE STATISTICS
====================================================
   Raw Data            : 0 records
   Features            : 0 records
   Predictions         : 0 records
   Models              : 0 records
====================================================

✅ MongoDB connection closed
```

❌ If you get errors:
- Check MongoDB URI in .env
- Verify MongoDB Atlas cluster is running
- Check network access settings (0.0.0.0/0)

---

### **Step 6: Upload CSV to MongoDB** 📤

```powershell
python upload_to_mongodb.py
```

✅ Expected output:
```
📤 UPLOADING DATA TO MONGODB
====================================================
📂 Loading CSV file...
✅ Loaded dataset:
   Records: 1,848
   Features: 66
   Date range: ...

🔌 Connecting to MongoDB...
✅ Connected to MongoDB database: aqi_karachi

💾 Uploading to MongoDB...
✅ Inserted 1,848 feature records

✅ Verifying upload...
====================================================
📊 DATABASE STATISTICS
====================================================
   Raw Data            : 0 records
   Features            : 1,848 records  ← DATA IS HERE!
   Predictions         : 0 records
   Models              : 0 records
====================================================

✅ UPLOAD COMPLETE!
```

---

## 🎉 SETUP COMPLETE!

You now have:
- ✅ VSCode project structure
- ✅ MongoDB Atlas configured
- ✅ Data uploaded to MongoDB
- ✅ Dependencies installed
- ✅ Configuration ready

---

## 🚀 NEXT STEPS

### **Immediate Next Steps:**

1. **Verify your data in MongoDB**
   - Go to MongoDB Atlas
   - Click: Browse Collections
   - You should see `features` collection with your data

2. **Next Session: Implement Feature Pipeline**
   - Edit `src/feature_pipeline.py`
   - Fetch hourly data from Open-Meteo
   - Engineer features
   - Update MongoDB

3. **After that: Implement Training Pipeline**
   - Edit `src/training_pipeline.py`
   - Train multiple models
   - Save best model
   - Evaluate performance

4. **Finally: Create Streamlit Dashboard**
   - Edit `app.py`
   - Display predictions
   - Show visualizations

---

## 📝 Troubleshooting

### **Problem: Can't activate venv**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### **Problem: MongoDB connection fails**
- Check internet connection
- Verify MongoDB URI in .env
- Check MongoDB Atlas is running
- Verify network access (0.0.0.0/0)

### **Problem: CSV not found**
- Make sure CSV is in `data/` folder
- Check filename: `karachi_complete_dataset.csv`
- No spaces or special characters in filename

### **Problem: Import errors**
- Make sure venv is activated
- Reinstall: `pip install -r requirements.txt`

---

## 🎯 Project Timeline

- ✅ **Week 1**: Setup (DONE!)
- ⏳ **Week 2**: Feature Pipeline + Training Pipeline
- ⏳ **Week 3**: GitHub Actions + Streamlit Dashboard
- ⏳ **Week 4**: Testing + Documentation + Report

---

## 📧 Need Help?

If you encounter any issues:
1. Check error messages carefully
2. Verify all credentials in .env
3. Check MongoDB Atlas dashboard
4. Review this guide step-by-step

---

**🎉 Congratulations! Your project foundation is ready!**

Next: Implement the feature pipeline and training pipeline.
