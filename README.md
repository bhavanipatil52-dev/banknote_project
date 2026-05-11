# Banknote Authentication - Model Stealing Attack Demo

## Project Overview
This project demonstrates a full stack machine learning application that classifies banknotes as real or fake, and simulates a black-box model stealing attack on it.

## Project Structure
## Dataset
- **Name:** Banknote Authentication Dataset
- **Source:** UCI Machine Learning Repository
- **Rows:** 1372
- **Features:** Variance, Skewness, Curtosis, Entropy
- **Target:** 0 = Real, 1 = Fake

## How to Run

### Step 1 - Train the model (once only)
```bash
python model/train_model.py
```

### Step 2 - Start the Flask API
```bash
python api/app.py
```

### Step 3 - Launch the Streamlit frontend
```bash
streamlit run frontend/frontend.py
```

### Step 4 - Run the attack (optional)
```bash
python attack/attack.py
```

## Results
- Original Model Accuracy: **99.6%**
- Copycat Model Agreement Rate: **93.67%**
- Attack Result: **HIGH RISK - Attack Successful**

## Technologies Used
- Python 3.14
- scikit-learn (Random Forest)
- Flask (REST API)
- Streamlit (Frontend)
- pandas, numpy (Data handling)
- joblib (Model saving)

## What is Model Stealing?
A black-box model stealing attack is where an attacker queries a public API repeatedly, collects the input-output pairs, and trains a copycat model that mimics the original — without ever accessing the training data or model file.