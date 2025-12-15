import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_dataset(file_path):
    """
    Memuat dataset dari file CSV
    
    Parameters:
    -----------
    file_path : str
        Path ke file CSV dataset
    
    Returns:
    --------
    df : DataFrame
        Dataset yang telah dimuat
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Dataset loaded successfully!")
        print(f"Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def handle_missing_values(df):
    """
    Menangani missing values dalam dataset
    
    Parameters:
    -----------
    df : DataFrame
        Dataset input
    
    Returns:
    --------
    df : DataFrame
        Dataset tanpa missing values
    """
    print("\n=== Handling Missing Values ===")
    missing_before = df.isnull().sum().sum()
    print(f"Missing values sebelum: {missing_before}")
    
    df_cleaned = df.dropna()
    missing_after = df_cleaned.isnull().sum().sum()
    print(f"Missing values setelah: {missing_after}")
    
    return df_cleaned

def handle_duplicates(df):
    """
    Menghapus data duplikat
    
    Parameters:
    -----------
    df : DataFrame
        Dataset input
    
    Returns:
    --------
    df : DataFrame
        Dataset tanpa duplikasi
    """
    print("\n=== Handling Duplicates ===")
    duplicates_before = df.duplicated().sum()
    print(f"Duplikasi sebelum: {duplicates_before}")
    
    df_no_dup = df.drop_duplicates()
    duplicates_after = df_no_dup.duplicated().sum()
    print(f"Duplikasi setelah: {duplicates_after}")
    print(f"Shape setelah cleaning: {df_no_dup.shape}")
    
    return df_no_dup

def encode_target_variable(df, target_col='Heart Disease', mapping={'Presence': 1, 'Absence': 0}):
    """
    Encoding kolom target dari kategorikal ke numerik
    
    Parameters:
    -----------
    df : DataFrame
        Dataset input
    target_col : str
        Nama kolom target
    mapping : dict
        Dictionary untuk mapping nilai kategorikal ke numerik
    
    Returns:
    --------
    df : DataFrame
        Dataset dengan target yang sudah di-encode
    """
    print("\n=== Encoding Target Variable ===")
    print(f"Nilai unik {target_col} sebelum encoding: {df[target_col].unique()}")
    
    if df[target_col].dtype == 'object':
        df[target_col] = df[target_col].map(mapping)
        print(f"Nilai unik {target_col} setelah encoding: {df[target_col].unique()}")
        print(f"Distribusi {target_col}:\n{df[target_col].value_counts()}")
    
    return df

def detect_outliers(df, target_col='Heart Disease'):
    """
    Mendeteksi outlier menggunakan metode IQR
    
    Parameters:
    -----------
    df : DataFrame
        Dataset input
    target_col : str
        Nama kolom target yang akan di-exclude
    
    Returns:
    --------
    outliers_count : Series
        Jumlah outlier per kolom
    """
    print("\n=== Deteksi Outlier (IQR Method) ===")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    outliers_count = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        outliers_count[col] = len(outliers)
    
    outliers_series = pd.Series(outliers_count).sort_values(ascending=False)
    print(outliers_series.head(10))
    
    return outliers_series

def split_features_target(df, target_col='Heart Disease'):
    """
    Memisahkan fitur dan target
    
    Parameters:
    -----------
    df : DataFrame
        Dataset input
    target_col : str
        Nama kolom target
    
    Returns:
    --------
    X : DataFrame
        Fitur (features)
    y : Series
        Target
    """
    print("\n=== Data Split (Features & Target) ===")
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    return X, y

def standardize_features(X):
    """
    Standarisasi fitur menggunakan StandardScaler
    
    Parameters:
    -----------
    X : DataFrame
        Fitur yang akan distandarisasi
    
    Returns:
    --------
    X_scaled : DataFrame
        Fitur yang telah distandarisasi
    scaler : StandardScaler
        Scaler object untuk transformasi data baru
    """
    print("\n=== Standarisasi Fitur ===")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    print("✅ Standarisasi selesai")
    print(X_scaled.head())
    
    return X_scaled, scaler

def split_train_test(X, y, test_size=0.2, random_state=42):
    """
    Membagi data menjadi training dan testing set
    
    Parameters:
    -----------
    X : DataFrame
        Fitur
    y : Series
        Target
    test_size : float
        Proporsi data testing (default: 0.2)
    random_state : int
        Random seed untuk reproducibility
    
    Returns:
    --------
    X_train, X_test, y_train, y_test : tuple
        Data training dan testing
    """
    print("\n=== Train-Test Split ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    print(f"Training target distribution:\n{y_train.value_counts()}")
    print(f"Testing target distribution:\n{y_test.value_counts()}")
    
    return X_train, X_test, y_train, y_test

def preprocess_pipeline(file_path, target_col='Heart Disease', 
                        encoding_map={'Presence': 1, 'Absence': 0},
                        test_size=0.2, random_state=42):
    """
    Pipeline lengkap untuk preprocessing data Heart Disease
    
    Parameters:
    -----------
    file_path : str
        Path ke file CSV dataset
    target_col : str
        Nama kolom target
    encoding_map : dict
        Dictionary untuk mapping target
    test_size : float
        Proporsi data testing
    random_state : int
        Random seed
    
    Returns:
    --------
    X_train, X_test, y_train, y_test : tuple
        Data yang siap untuk training
    scaler : StandardScaler
        Scaler object untuk transformasi data baru
    """
    print("="*60)
    print("🚀 MEMULAI PREPROCESSING PIPELINE")
    print("="*60)
    
    # 1. Load dataset
    df = load_dataset(file_path)
    if df is None:
        return None
    
    # 2. Handle missing values
    df = handle_missing_values(df)
    
    # 3. Handle duplicates
    df = handle_duplicates(df)
    
    # 4. Encode target variable
    df = encode_target_variable(df, target_col, encoding_map)
    
    # 5. Detect outliers (informasi saja, tidak dihapus)
    _ = detect_outliers(df, target_col)
    
    # 6. Split features and target
    X, y = split_features_target(df, target_col)
    
    # 7. Standardize features
    X_scaled, scaler = standardize_features(X)
    
    # 8. Split train-test
    X_train, X_test, y_train, y_test = split_train_test(
        X_scaled, y, test_size, random_state
    )
    
    print("\n" + "="*60)
    print("✅ PREPROCESSING SELESAI - DATA SIAP UNTUK TRAINING!")
    print("="*60)
    
    return X_train, X_test, y_train, y_test, scaler

# Contoh penggunaan
if __name__ == "__main__":
    # Path ke dataset
    file_path = '../Heart_Disease_Prediction.csv'
    
    # Jalankan preprocessing pipeline
    result = preprocess_pipeline(file_path)
    
    if result is not None:
        X_train, X_test, y_train, y_test, scaler = result
        
        print("\n" + "="*60)
        print("📊 RINGKASAN DATA YANG SIAP DILATIH:")
        print("="*60)
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        print(f"Scaler: {type(scaler).__name__}")
