# 🚀 Streamlit Cloud Deployment Guide

## Prerequisites
✅ GitHub repository: https://github.com/Azhism/karachi-aqi-predictor
✅ MongoDB Atlas cluster with data
✅ MongoDB secret added to GitHub Actions

---

## Step-by-Step Deployment

### 1. Create Streamlit Cloud Account
1. Go to: https://streamlit.io/cloud
2. Click **Sign up** (use your GitHub account for easy integration)
3. Click **Authorize Streamlit**

### 2. Deploy Your App
1. Click **New app** button
2. Fill in the deployment form:
   - **Repository:** `Azhism/karachi-aqi-predictor`
   - **Branch:** `main`
   - **Main file path:** `app.py`
3. Click **Advanced settings** (before deploying)

### 3. Configure Secrets
In the **Secrets** section, paste this (replace with your values):

```toml
MONGODB_URI = "mongodb+srv://azhism:<YOUR_PASSWORD>@cluster0.ofp1i.mongodb.net/"
MONGODB_DATABASE = "aqi_karachi"
CITY_NAME = "Karachi"
LATITUDE = "24.8608"
LONGITUDE = "67.0104"
PREDICTION_HORIZON = "72"
```

**Important:** Replace `<YOUR_PASSWORD>` with your actual MongoDB password!

### 4. Deploy!
1. Click **Deploy!**
2. Wait 2-3 minutes for build to complete
3. Your app will be live at: `https://<your-app-name>.streamlit.app`

---

## 🎯 Your Live Dashboard Will Show:

### Features:
- 📊 **Real-time AQI Predictions** (24h, 48h, 72h ahead)
- 📈 **Historical Trends** (Last 7 days)
- 🤖 **Model Comparison** (RandomForest, XGBoost, LightGBM)
- 🎨 **Color-coded AQI Categories** (Good, Moderate, Unhealthy, etc.)
- 📉 **Interactive Charts** with Plotly

### Automatic Updates:
- Every hour: Fresh data collection
- Every day: Model retraining
- Always: Latest predictions

---

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution:** Make sure `requirements.txt` is in the root directory

### Issue: "Connection Error to MongoDB"
**Solution:** 
1. Check your MongoDB Atlas IP whitelist (should allow `0.0.0.0/0` for Streamlit Cloud)
2. Verify `MONGODB_URI` secret is correct
3. Check MongoDB Atlas user has read/write permissions

### Issue: "No data found"
**Solution:** 
1. Make sure hourly data collection workflow ran at least once
2. Check GitHub Actions: https://github.com/Azhism/karachi-aqi-predictor/actions
3. Verify MongoDB has data in `features` collection

### Issue: App is slow
**Solution:** 
- Streamlit Cloud free tier has limited resources
- Consider upgrading to Streamlit Cloud Pro if needed
- Check MongoDB Atlas tier (M0 free tier may be slow)

---

## 📱 Sharing Your Dashboard

Once deployed, share your app URL:
- 🌐 **Public URL:** `https://<your-app-name>.streamlit.app`
- 📱 **Mobile friendly:** Works on any device
- 🔗 **Embed:** Can be embedded in websites

---

## 🔄 Updating Your App

Any code changes pushed to GitHub `main` branch will:
1. Automatically trigger Streamlit Cloud rebuild
2. Update your live app within 2-3 minutes
3. No manual redeployment needed!

---

## 💡 Tips

- **Monitor Usage:** Check Streamlit Cloud dashboard for app analytics
- **View Logs:** Click "Manage app" → "Logs" to debug issues
- **Reboot App:** If stuck, click "Reboot app" in Streamlit Cloud dashboard
- **Performance:** Free tier sleeps after inactivity; first load may be slow

---

## 🎉 Success Checklist

- ✅ App deployed and accessible
- ✅ Predictions showing 3 time horizons
- ✅ Historical chart displays data
- ✅ Model comparison shows accuracy metrics
- ✅ No errors in logs
- ✅ Hourly data collection running
- ✅ Daily model retraining scheduled

---

**Need help?** Check Streamlit docs: https://docs.streamlit.io/streamlit-community-cloud
