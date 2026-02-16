# 🚀 QUICK REFERENCE - Common Commands

## 📦 Environment Setup

### Create and activate virtual environment
```powershell
# Create venv
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# If execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Deactivate
deactivate
```

### Install dependencies
```powershell
pip install -r requirements.txt
```

---

## 🧪 Testing

### Test complete setup
```powershell
python test_setup.py
```

### Test MongoDB connection only
```powershell
python src/database.py
```

---

## 📤 Data Management

### Upload CSV to MongoDB (first time)
```powershell
python upload_to_mongodb.py
```

### Check MongoDB collections
```powershell
python src/database.py
```

---

## 🔄 Pipeline Operations

### Run feature pipeline (when implemented)
```powershell
python src/feature_pipeline.py
```

### Run training pipeline (when implemented)
```powershell
python src/training_pipeline.py
```

---

## 🌐 Streamlit Dashboard

### Run locally
```powershell
streamlit run app.py
```

### Run on specific port
```powershell
streamlit run app.py --server.port 8501
```

---

## 🐙 Git Commands

### Initialize repository
```powershell
git init
git add .
git commit -m "Initial commit"
```

### Connect to GitHub
```powershell
git remote add origin https://github.com/yourusername/karachi-aqi-predictor.git
git branch -M main
git push -u origin main
```

### Regular commits
```powershell
git add .
git commit -m "Your message"
git push
```

---

## 📊 MongoDB Atlas Dashboard

### View your data
1. Go to: https://cloud.mongodb.com
2. Click: Database → Browse Collections
3. Database: `aqi_karachi`
4. Collections:
   - `features` - Your engineered features
   - `predictions` - Model predictions
   - `models` - Model metadata

---

## 🔍 Useful Python Commands

### Check package versions
```powershell
pip list
```

### Update a package
```powershell
pip install --upgrade package_name
```

### Freeze dependencies
```powershell
pip freeze > requirements.txt
```

---

## 📝 File Locations

### Configuration
- `.env` - Environment variables (MongoDB URI, etc.)
- `src/config.py` - Project configuration

### Data
- `data/` - Place CSV files here
- MongoDB - Online feature store

### Models
- `models/` - Saved model files (.pkl)
- MongoDB `models` collection - Model metadata

### Code
- `src/database.py` - MongoDB operations
- `src/feature_pipeline.py` - Feature engineering
- `src/training_pipeline.py` - Model training
- `app.py` - Streamlit dashboard

---

## ⚡ Quick Workflow

### Daily Development
```powershell
# 1. Activate environment
.\venv\Scripts\Activate.ps1

# 2. Run tests
python test_setup.py

# 3. Work on your code
# Edit files in VSCode

# 4. Test locally
python src/your_script.py

# 5. Commit changes
git add .
git commit -m "Description"
git push
```

---

## 🆘 Troubleshooting

### Virtual environment not activating
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### MongoDB connection fails
1. Check `.env` file has correct URI
2. Verify MongoDB Atlas cluster is running
3. Check Network Access allows your IP
4. Test connection: `python src/database.py`

### Import errors
```powershell
# Make sure venv is activated
.\venv\Scripts\Activate.ps1

# Reinstall packages
pip install -r requirements.txt
```

### CSV not found
- Place CSV in `data/` folder
- Filename should be: `karachi_complete_dataset.csv`

---

## 📋 Project Checklist

### Setup Phase (Week 1)
- [x] Create project structure
- [x] Setup MongoDB Atlas
- [x] Configure .env file
- [x] Install dependencies
- [x] Upload CSV to MongoDB
- [x] Test connection

### Development Phase (Week 2)
- [ ] Implement feature_pipeline.py
- [ ] Test feature pipeline locally
- [ ] Implement training_pipeline.py
- [ ] Train initial models
- [ ] Evaluate model performance

### Automation Phase (Week 3)
- [ ] Create GitHub Actions workflows
- [ ] Test CI/CD pipeline
- [ ] Implement Streamlit dashboard
- [ ] Deploy to Streamlit Cloud

### Finalization (Week 4)
- [ ] Complete documentation
- [ ] Write project report
- [ ] Create presentation
- [ ] Submit project

---

## 🎯 Key Files to Remember

**Never commit to Git:**
- `.env` (contains secrets)
- `venv/` (virtual environment)
- `data/*.csv` (large data files)

**Always commit to Git:**
- `src/` (your code)
- `requirements.txt`
- `.gitignore`
- `README.md`
- `app.py`

---

## 💡 Helpful Tips

1. **Always activate venv** before running any Python command
2. **Test frequently** with `python test_setup.py`
3. **Check MongoDB Atlas** to verify data uploads
4. **Commit often** with meaningful messages
5. **Document as you go** - update README.md

---

## 📚 Resources

- MongoDB Atlas: https://cloud.mongodb.com
- Streamlit Docs: https://docs.streamlit.io
- Scikit-learn Docs: https://scikit-learn.org
- Open-Meteo API: https://open-meteo.com

---

**Need help? Review SETUP_GUIDE.md for detailed instructions!**
