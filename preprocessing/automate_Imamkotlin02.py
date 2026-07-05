# preprocessing/automate_Nama-siswa.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_preprocessing():
    print("=== Memulai Pipa Pemrosesan Data ===")
    
    # 1. Data Loading
    raw_data_path = '../heart_disease_raw/heart.csv'
    if not os.path.exists(raw_data_path):
        # Antisipasi jika dijalankan dari root repositori saat GitHub Actions
        raw_data_path = 'heart_disease_raw/heart.csv'
        output_dir = 'preprocessing/heart_disease_preprocessing'
    else:
        output_dir = 'heart_disease_preprocessing'
        
    df = pd.read_csv(raw_data_path)
    print(f"Data mentah berhasil dimuat. Ukuran: {df.shape}")

    # 2. Preprocessing (Pisahkan Fitur dan Target)
    # Asumsi kolom target bernama 'target'
    X = df.drop(columns=['target'])
    y = df['target']

    # Split data menjadi Train dan Test (80:20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling menggunakan StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Konversi kembali ke DataFrame agar rapi saat disimpan
    X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)

    # 3. Export Data Siap Latih
    os.makedirs(output_dir, exist_ok=True)
    
    X_train_df.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test_df.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    print(f"Pipa selesai! Data hasil preprocessing disimpan di: {output_dir}")

if __name__ == "__main__":
    run_preprocessing()
