# Eksperimen_SML_Raifal_Bagus_Afdiansah

Eksperimen Machine Learning untuk prediksi penyakit jantung dengan preprocessing otomatis menggunakan GitHub Actions.

## 📁 Struktur Folder

```
Eksperimen_SML_Raifal_Bagus_Afdiansah/
├── .github/
│   └── workflows/
│       └── preprocessing.yml          # GitHub Actions workflow
├── Heart_Disease_Prediction.csv       # Dataset raw
├── preprocessing/
│   ├── Eksperimen_Raifal_Bagus_Afdiansah.ipynb
│   ├── automate_Raifal_Bagus_Afdiansah.py
│   └── Heart_Disease_Preprocessing.csv  # Dataset hasil preprocessing
└── README.md
```

## 🚀 Fitur

- ✅ **Preprocessing Otomatis** dengan GitHub Actions
- ✅ **Data Cleaning** (Missing Values & Duplicates)
- ✅ **Target Encoding** (Presence/Absence → 1/0)
- ✅ **Outlier Detection** dengan IQR Method
- ✅ **Feature Standardization** dengan StandardScaler
- ✅ **Stratified Train-Test Split** (80:20)

## 📊 Dataset

**Sumber**: Heart Disease Prediction Dataset  
**Jumlah Data**: 270 pasien  
**Fitur**: 13 fitur medis  
**Target**: Heart Disease (Presence/Absence)

### Fitur Dataset:
- Age
- Sex
- Chest pain type
- BP (Blood Pressure)
- Cholesterol
- FBS over 120
- EKG results
- Max HR
- Exercise angina
- ST depression
- Slope of ST
- Number of vessels fluro
- Thallium

## 🔄 GitHub Actions Workflow

Workflow akan otomatis berjalan ketika:
1. **Push** ke branch `main`/`master` yang mengubah:
   - `Heart_Disease_Prediction.csv`
   - `preprocessing/automate_Heart_Disease_Prediction.py`
2. **Pull Request** ke branch `main`/`master`
3. **Manual Trigger** melalui tab Actions

### Output Workflow:
- File `Heart_Disease_Preprocessing.csv` di folder `preprocessing/`
- Artifact yang dapat diunduh (tersimpan 30 hari)

## 💻 Cara Penggunaan

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/Eksperimen_SML_Raifal_Bagus_Afdiansah.git
cd Eksperimen_SML_Raifal_Bagus_Afdiansah
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 3. Jalankan Preprocessing Manual
```bash
cd preprocessing
python automate_Heart_Disease_Prediction.py
```

### 4. Gunakan dalam Python
```python
from preprocessing.automate_Heart_Disease_Prediction import preprocess_pipeline

# Preprocessing otomatis
X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(
    'Heart_Disease_Prediction.csv',
    save_output=True
)
```

## 📈 Hasil Preprocessing

| Komponen | Shape | Deskripsi |
|----------|-------|-----------|
| X_train | (216, 13) | Data training (80%) |
| X_test | (54, 13) | Data testing (20%) |
| y_train | (216,) | Label training |
| y_test | (54,) | Label testing |

**Distribusi Kelas:**
- Train: 120 (Absence) vs 96 (Presence)
- Test: 30 (Absence) vs 24 (Presence)

## 🛠️ Teknologi

- **Python 3.10**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning
- **GitHub Actions** - CI/CD automation

## 👨‍💻 Author

**Raifal Bagus Afdiansah**  
Eksperimen Supervised Machine Learning - Semester 7

## 📝 License

This project is for educational purposes.
