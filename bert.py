
import pandas as pd
import numpy as np
import torch
from transformers import pipeline
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import re

# --- 1. VERİ YÜKLEME ---
path = r'C:\Users\tumer\OneDrive\Masaüstü\e-ticaret\data\\'
try:
    reviews = pd.read_csv(path + 'olist_order_reviews_dataset.csv')
    print("Veri yüklendi.")
except FileNotFoundError:
    print("Veri dosyası bulunamadı, lütfen yolu kontrol edin.")
    exit()

# Sadece metin içeren yorumları filtrele
df_reviews = reviews[['review_id', 'review_comment_message', 'review_score']].dropna().copy()
df_reviews['cleaned_text'] = df_reviews['review_comment_message'].str.lower()

# --- 2. BERT MODEL KURULUMU (GPU DESTEKLİ VE HATASIZ) ---
device = 0 if torch.cuda.is_available() else -1
print(f"Kullanılan Cihaz: {'GPU' if device == 0 else 'CPU'}")

# Portekizceye özel Bertweet modeli
model_name = "pysentimiento/bertweet-pt-sentiment"

# Truncation=True: Uzun metinlerdeki 'Index out of range' hatasını çözer
sentiment_task = pipeline(
    "sentiment-analysis", 
    model=model_name, 
    device=device,
    truncation=True,
    max_length=128
)

# --- 3. BATCH PROCESSING (TOPLU İŞLEME) ---
comments = df_reviews['review_comment_message'].astype(str).tolist()
batch_size = 32
results = []

print("BERT Analizi Başlıyor... Bu işlem veri boyutuna göre vakit alabilir.")
for i in tqdm(range(0, len(comments), batch_size)):
    batch = comments[i : i + batch_size]
    # BERT limitine takılmamak için karakter seviyesinde ön kesme
    batch = [text[:512] for text in batch] 
    batch_results = sentiment_task(batch)
    results.extend(batch_results)

# Sonuçları ana tabloya işle
df_reviews['BERT_Sentiment'] = [res['label'] for res in results]
df_reviews['BERT_Confidence'] = [res['score'] for res in results]

# --- 4. GÖRSELLEŞTİRME 1: PUAN VS DUYGU (ISI HARİTASI) ---
plt.figure(figsize=(10, 6))
pivot_table = pd.crosstab(df_reviews['review_score'], df_reviews['BERT_Sentiment'], normalize='index')
sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title('Müşteri Puanı ile BERT Duygu Tahminleri Arasındaki İlişki')
plt.xlabel('BERT Tahmini (NEG: Negatif, NEU: Nötr, POS: Pozitif)')
plt.ylabel('Müşterinin Verdiği Puan')
plt.show()

# --- 5. GÖRSELLEŞTİRME 2: GİZLİ MEMNUNİYETSİZLER ---
# 5 puan verip BERT'in 'NEG' (Negatif) dediği riskli grup
hidden_risk = df_reviews[(df_reviews['review_score'] == 5) & (df_reviews['BERT_Sentiment'] == 'NEG')]

print(f"\nAnaliz Sonucu:")
print(f"Toplam Analiz Edilen Yorum: {len(df_reviews)}")
print(f"5 Puan Verip Metni Negatif Olan (Gizli Risk) Sayısı: {len(hidden_risk)}")

if not hidden_risk.empty:
    plt.figure(figsize=(10, 5))
    # Seaborn FutureWarning düzenlemesi: hue ve legend eklendi
    sns.countplot(data=df_reviews, x='review_score', hue='BERT_Sentiment', palette='viridis')
    plt.title('Puan Dağılımına Göre BERT Duygu Sınıflandırması')
    plt.legend(title='BERT Duygusu', loc='upper left')
    plt.show()

# --- 6. SONUÇLARI KAYDETME ---
df_reviews.to_csv('bert_sentiment_sonuclari.csv', index=False)
print("Sonuçlar 'bert_sentiment_sonuclari.csv' olarak kaydedildi.")