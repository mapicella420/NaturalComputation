import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Carica solo i primi 10.000 record del dataset
df = pd.read_csv("human_vital_signs_dataset_2024.csv")

# 2. Mappa la colonna Risk Category in valori numerici
df["risk_category"] = df["Risk Category"].map({"Low Risk": 0, "High Risk": 1})

# 3. Rimuovi la colonna testuale originale se vuoi
df.drop(columns=["Risk Category"], inplace=True)

# 4. Suddividi in train/test (80/20 stratificato sulla label)
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["risk_category"], random_state=42)

# 5. Salva i file
train_df.to_csv("human_vital_signs_train.csv", index=False)
test_df.to_csv("human_vital_signs_test.csv", index=False)

print("✅ Split completato:")
print(f" - Train set: {len(train_df)} righe")
print(f" - Test set: {len(test_df)} righe")
