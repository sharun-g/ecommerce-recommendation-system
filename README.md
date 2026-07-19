# 🛒 E-commerce Product Recommendation System

## 📌 Overview
This project implements a **Hybrid Product Recommendation System** that provides personalized and relevant product suggestions in an e-commerce environment. It combines **Collaborative Filtering** and **FP-Growth (Association Rule Mining)** to improve recommendation quality.

---

## 🚀 Features
- Personalized recommendations using Collaborative Filtering  
- Product association using FP-Growth algorithm  
- Hybrid model combining both approaches  
- Top-N product recommendations  
- Handles invalid inputs (product not found / not in model)  

---

## 🧠 Models Used

### 🔹 Collaborative Filtering
- Uses cosine similarity  
- Captures user preferences  

### 🔹 FP-Growth
- Finds frequent itemsets  
- Generates association rules  

### 🔹 Hybrid Model
- Combines CF + FP-Growth using weighted scoring  
- Improves recommendation accuracy and diversity  

---

## 🗂️ Project Structure

```text
├── data/
│   └── dataset                 # Raw dataset
│
├── Reports/
│   ├── DWDM project report.pptx         
│   ├── final report.docx 
|
├── src/
│   ├── data_loader.py         # Load dataset
│   ├── preprocess.py          # Data preprocessing
│   ├── cf_model.py            # Collaborative Filtering
│   ├── fp_growth_model.py     # FP-Growth implementation
│   └── hybrid_model.py        # Hybrid recommendation logic
│
├── EDA.py                     # Exploratory Data Analysis
├── train.py                   # Model training
├── evaluation.py              # Model evaluation
├── visualization.py           # Graphs and plots
├── main.py                    # Run recommendation system
├── requirements.txt           # Dependencies
│── Dockerfile
│── start.sh
├── README.md
├── LICENSE
└── .gitignore
```

---

## ⚙️ Installation & Run

### 1. Clone the repository
```bash
git clone https://github.com/sharun-g/ecommerce-recommendation-system.git
cd ecommerce-recommendation-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the recommendation models
```bash
python train.py
```

### 4. Run the recommendation system
```bash
python main.py
```
## Docker

Build the Docker image

```bash
docker build -t ecommerce-recommendation-system .
```

Run the container

```bash
docker run -it --rm ecommerce-recommendation-system
```

## Sample Usage

```
Enter product name:  Organic Strawberries

Recommended Products

• Organic Baby Spinach
• Banana
• Organic Hass Avocado
```
## Author

**Sharun**
