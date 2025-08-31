# LeafLens 




# 🌱 Plant Disease Detection using CNN

This project detects **plant diseases** from leaf images using **Convolutional Neural Networks (CNNs)**.
It provides detailed information about diseases, fertilizers, precautions, and growth periods, helping farmers and agriculturists manage crop health effectively.

---

## 📊 Dataset

We use the **PlantVillage Dataset** from Kaggle:
🔗 [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)

It contains **54,000+ images** of healthy and diseased plant leaves across **14 crops** and **38 disease classes**.

---

## 🚀 Features

* 🌿 **Plant Disease Classification** using CNN
* 📊 **Training & Testing Notebooks** (`Train_plant_disease.ipynb`, `Test_plant_disease.ipynb`)
* 💊 **Fertilizer & Precaution Suggestions** (CSV knowledge base)
* ⏳ **Growth Period Insights** for plants
* 📱 **App Interface** (`main.py`) to run predictions on new images
* 🎥 **Demo Video** (`Output/Output.mp4`)

---

## 🛠️ Tech Stack

* **Deep Learning:** TensorFlow / Keras
* **Data Handling:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Frontend/App:** Python (Streamlit / Tkinter depending on script)

---

## 📂 Project Structure

```
PlantDisease/
│── App/
│   ├── main.py                        # App entry script
│   ├── Train_plant_disease.ipynb      # Training notebook
│   ├── Test_plant_disease.ipynb       # Testing notebook
│   ├── plant_disease_details.csv      # Disease descriptions
│   ├── plant_diseases_fertilizers_precautions.csv
│   ├── extended_plant_diseases_fertilizers_precautions.csv
│   ├── extended_plant_diseases_with_growth_periods.csv
│   ├── training_hist.json             # Training history logs
│   ├── home_page.jpeg, image.png      # UI assets
│── Output/
│   ├── Output.mp4                     # Demo video
│── README.md
```

---

## ⚙️ Installation & Setup

1. **Clone the repo**

```bash
git clone https://github.com/yourusername/PlantDiseaseDetection.git
cd PlantDisease/App
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Download dataset**
   Get the PlantVillage dataset from Kaggle and place it inside a `dataset/` folder.

4. **Train model**

```bash
jupyter notebook Train_plant_disease.ipynb
```

5. **Run app**

```bash
python main.py
```

---

## ▶️ Usage

* Upload a plant leaf image
* Model predicts the **disease**
* App shows **fertilizer suggestions & precautions**

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📜 License

MIT License – Feel free to use & modify.

---
