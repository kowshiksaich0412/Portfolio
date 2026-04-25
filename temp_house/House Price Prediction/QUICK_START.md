# House Price Prediction - Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model
```bash
python main.py
```

This will:
- ✓ Load the dataset
- ✓ Preprocess and explore data
- ✓ Train 3 models (Linear Regression, Decision Tree, Random Forest)
- ✓ Evaluate models with R² and RMSE scores
- ✓ Display feature importance
- ✓ Save the best model

**Expected time**: 1-2 minutes

### Step 3: Start the Web App
```bash
python app.py
```

Then open your browser to: **http://localhost:5000**

---

## 📋 Input Fields Reference

| Field | Range | Type | Example |
|-------|-------|------|---------|
| Area | 100-10,000 | Number | 2500 |
| Bedrooms | 1-10 | Number | 3 |
| Bathrooms | 1-10 | Number | 2.5 |
| Floors | 1-5 | Number | 2 |
| Year Built | 1800-2024 | Number | 2000 |
| Location | Dropdown | String | Suburban |
| Condition | Dropdown | String | Good |
| Garage | Dropdown | String | Yes |

---

## 📊 What You'll See

### During Training (main.py)
- Dataset statistics
- Missing value counts
- Categorical encoding mappings
- Correlation heatmap
- Feature distribution plots
- Model metrics comparison (R², RMSE)
- Feature importance charts

### After Training
- Generated files:
  - `eda_analysis.png` - EDA visualizations
  - `feature_importance.png` - Feature importance charts
  - `models/best_model.pkl` - Trained model
  - `models/scaler.pkl` - Feature scaler
  - `models/label_encoders.pkl` - Categorical encoders
  - `models/feature_names.pkl` - Feature names
  - `models/all_models.pkl` - All 3 models

### During Prediction (app.py)
- Beautiful web interface
- Real-time predictions
- Formatted price output
- Error handling

---

## 🔄 Typical Workflow

1. **First time?** Start with Step 1 (Install) → Step 2 (Train) → Step 3 (Run Web App)

2. **Subsequent runs?** Just run Step 3 (Web App) since model is already trained

3. **Retrain model?** Run Step 2 again if you want to retrain

---

## 💻 System Requirements

- Python 3.8+
- 4GB RAM (minimum)
- 500MB disk space

---

## ❓ FAQ

**Q: Do I need to rerun main.py every time?**
A: No, just run `python app.py` to use the saved model.

**Q: What if I get "Model not found" error?**
A: Run `python main.py` first to train the model.

**Q: Can I modify the dataset?**
A: Yes! Just replace `House Price Prediction Dataset.csv` with your data and rerun `python main.py`

**Q: Which model performs best?**
A: Check the R² score - higher is better. The best model is automatically selected and saved.

**Q: How accurate are predictions?**
A: Check the RMSE value from the training output. Lower RMSE = more accurate.

---

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| Module not found | Run: `pip install -r requirements.txt` |
| Model not loaded | Run: `python main.py` |
| Port 5000 in use | Change port in app.py or close other apps |
| Permission denied | Run with administrator privileges |
| Slow training | This is normal - can take 1-2 minutes |

---

## 📞 Support

Check the `README.md` file for detailed documentation and troubleshooting.

---

**Happy predicting! 🎉**
