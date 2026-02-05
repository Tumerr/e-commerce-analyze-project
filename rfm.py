import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# --- VERİ YÜKLEME VE BİRLEŞTİRME ---
# Güncel dosya yolu (Hata olasılığına karşı raw string (r) olarak tanımlandı)
path = r'C:\Users\tumer\OneDrive\Masaüstü\e-ticaret\data\\' 

try:
    # CSV dosyalarını okuma
    customers = pd.read_csv(path + 'olist_customers_dataset.csv')
    orders = pd.read_csv(path + 'olist_orders_dataset.csv')
    order_items = pd.read_csv(path + 'olist_order_items_dataset.csv')
    products = pd.read_csv(path + 'olist_products_dataset.csv')
    category_translation = pd.read_csv(path + 'product_category_name_translation.csv')

    # Tabloları Birleştirme
    df_master = pd.merge(orders, customers, on='customer_id', how='left')
    df_master = pd.merge(df_master, order_items, on='order_id', how='left')
    df_master = pd.merge(df_master, products, on='product_id', how='left')
    df_master = pd.merge(df_master, category_translation, 
                         on='product_category_name', how='left')
    
    # İsimlendirme ve Temizlik
    df_master.rename(columns={'product_category_name_english': 'product_category'}, inplace=True)
    df_master.drop('product_category_name', axis=1, inplace=True)
    
    # Tarih Dönüşümleri
    df_master['order_purchase_timestamp'] = pd.to_datetime(df_master['order_purchase_timestamp'], errors='coerce')
    
    # Eksik değerleri ve teslim edilmeyen siparişleri filtreleme
    df_master.dropna(subset=['product_id', 'customer_unique_id'], inplace=True)
    df_master = df_master[df_master['order_status'] == 'delivered'].copy() 
    
    # Toplam Harcama Hesaplama
    df_master['Total_Price'] = df_master['price'] + df_master['freight_value']
    
    print("Veri Hazırlığı Tamamlandı.")

except FileNotFoundError:
    print(f"HATA: Veri setleri {path} yolunda bulunamadı. Lütfen yolu kontrol edin.")
    exit()

# --- RFM METRİKLERİNİN HESAPLANMASI ---

# Analiz Referans Tarihi
last_date = df_master['order_purchase_timestamp'].max()
snapshot_date = last_date + timedelta(days=1) 

# RFM değerlerinin hesaplanması (Bu adım 'rfm_results_corrected'ı oluşturur!)
rfm_results_corrected = df_master.groupby('customer_unique_id').agg(
    Recency=('order_purchase_timestamp', lambda x: (snapshot_date - x.max()).days),
    Frequency=('order_id', 'nunique'),
    Monetary=('Total_Price', 'sum')
).reset_index()

print("RFM Metrikleri Hesaplandı.")
print(rfm_results_corrected.head())

# --- K-MEANS İLE RFM SEGMENTASYONU ---

# 1. Logaritmik Dönüşüm (Ölçek farklılıklarını azaltmak için)
rfm_log = rfm_results_corrected[['Recency', 'Frequency', 'Monetary']].copy()
# Monetary ve Frequency için log dönüşümü
rfm_log['Monetary'] = np.log(rfm_log['Monetary'] + 1)
rfm_log['Frequency'] = np.log(rfm_log['Frequency'] + 1)

# 2. Standardizasyon (Ölçekleme)
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)
rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=['R_scaled', 'F_scaled', 'M_scaled'])

# 3. K-Means Uygulama (Örnek olarak k=4 kullanıldı)
final_k = 4
kmeans_model = KMeans(n_clusters=final_k, random_state=42, n_init=10)
kmeans_model.fit(rfm_scaled_df)

# Küme etiketlerini ana tabloya ekleme
rfm_results_corrected['K_Cluster'] = kmeans_model.labels_

# 4. Küme Profili Çıkarma (Düzeltilmiş)
cluster_profile = rfm_results_corrected.groupby('K_Cluster').agg(
    Musteri_Sayisi=('customer_unique_id', 'count'),
    # Ort_Recency=('Recency', 'mean'), ARTIK YUVARLAMA BURADA DEĞİL
    Ort_Recency=('Recency', 'mean'), 
    Ort_Frequency=('Frequency', 'mean'),
    Ort_Monetary=('Monetary', 'mean')
).reset_index()

# YUVARLAMA İŞLEMİ BURAYA TAŞINDI: Agregasyon sonrası DataFrame üzerinde
cluster_profile['Ort_Recency'] = cluster_profile['Ort_Recency'].round(1)
cluster_profile['Ort_Frequency'] = cluster_profile['Ort_Frequency'].round(2)
cluster_profile['Ort_Monetary'] = cluster_profile['Ort_Monetary'].round(2)

cluster_profile['Yuzde'] = (cluster_profile['Musteri_Sayisi'] / cluster_profile['Musteri_Sayisi'].sum() * 100).round(2)

print("\n--- K-Means ile RFM Segmentasyon Sonuçları ---")
print(cluster_profile.sort_values(by='Ort_Recency', ascending=True))

cluster_profile = pd.DataFrame({
    'K_Cluster': [1, 0, 3, 2],
    'Musteri_Sayisi': [35695, 27855, 27007, 2801],
    'Ort_Recency': [147.8, 173.5, 220.3, 425.4],
    'Ort_Frequency': [1.00, 1.00, 2.11, 1.00],
    'Ort_Monetary': [69.05, 318.23, 308.53, 119.47],
    'Yuzde': [38.23, 29.84, 28.93, 3.00]
})

# Kümelere anlamlı isimler atama (Önceki yorumlamamızdan alınmıştır)
cluster_name_map = {
    1: '1. Potansiyel Müşteriler (Yeni/Düşük M)',
    0: '2. Büyük Harcamacılar (Tek Seferlik)',
    3: '3. Risk Altındakiler(Yüksek R)',
    2: '4. Sadıklar(Yüksek F & M)'
}
cluster_profile['Segment_Adi'] = cluster_profile['K_Cluster'].map(cluster_name_map)

plt.figure(figsize=(8, 4))
# Müşteri sayısına göre sıralama
data_sorted = cluster_profile.sort_values(by='Musteri_Sayisi', ascending=False)
sns.barplot(x='Segment_Adi', y='Musteri_Sayisi', data=data_sorted, palette="viridis")

# Veri etiketlerini ekleme
for index, row in data_sorted.iterrows():
    plt.text(index, row['Musteri_Sayisi'] + 500, f"{row['Musteri_Sayisi']}\n({row['Yuzde']}%)", color='black', ha="center")

plt.title('Segmentlere Göre Müşteri Sayısı Dağılımı (Büyüklük)', fontsize=15)
plt.xlabel('RFM Segmenti')
plt.ylabel('Müşteri Sayısı')
plt.xticks(rotation=10, ha='right')
plt.grid(axis='y', linestyle='--')
plt.show()

plt.figure(figsize=(10, 6))
# Recency'de düşük değer iyidir, bu yüzden düşük Recency'e göre sıralama (en iyi segment başta)
data_sorted = cluster_profile.sort_values(by='Ort_Recency', ascending=True)
sns.barplot(x='Segment_Adi', y='Ort_Recency', data=data_sorted, palette="magma")

# Veri etiketlerini ekleme
for index, row in data_sorted.iterrows():
    plt.text(index, row['Ort_Recency'] + 10, f"{row['Ort_Recency']} Gün", color='black', ha="center")

plt.title('Segmentlere Göre Ortalama Yenilik (Recency)', fontsize=15)
plt.xlabel('RFM Segmenti')
plt.ylabel('Ortalama Gün Farkı (Son Alışverişten Bu Yana)')
plt.xticks(rotation=15, ha='right')
plt.grid(axis='y', linestyle='--')
plt.show()

plt.figure(figsize=(10, 6))
# Frequency'e göre sıralama
data_sorted = cluster_profile.sort_values(by='Ort_Frequency', ascending=False)
sns.barplot(x='Segment_Adi', y='Ort_Frequency', data=data_sorted, palette="cividis")

# Veri etiketlerini ekleme
for index, row in data_sorted.iterrows():
    plt.text(index, row['Ort_Frequency'] + 0.05, f"{row['Ort_Frequency']:.2f}", color='black', ha="center")

plt.title('Segmentlere Göre Ortalama Sıklık (Frequency)', fontsize=15)
plt.xlabel('RFM Segmenti')
plt.ylabel('Ortalama Sipariş Sayısı')
plt.xticks(rotation=15, ha='right')
plt.grid(axis='y', linestyle='--')
plt.show()

plt.figure(figsize=(10, 7))
# Monetary değere göre sıralama
data_sorted = cluster_profile.sort_values(by='Ort_Monetary', ascending=False)
sns.barplot(x='Segment_Adi', y='Ort_Monetary', data=data_sorted, palette="plasma")

# Veri etiketlerini ekleme
for index, row in data_sorted.iterrows():
    plt.text(index, row['Ort_Monetary'] + 10, f"R$ {row['Ort_Monetary']:.2f}", color='black', ha="center")

plt.title('Segmentlere Göre Ortalama Parasal Değer (Monetary)', fontsize=15)
plt.xlabel('RFM Segmenti')
plt.ylabel('Ortalama Harcama (R$)')
plt.xticks(rotation=15, ha='right')
plt.grid(axis='y', linestyle='--')
plt.show()




import pandas as pd
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Tekrar kullanılacak olan df_master ve rfm_results_corrected'ın var olduğunu varsayıyoruz.
# Eğer kodunuzda bu değişkenler silinmişse, RFM analizi kodunuzun ilk kısımlarını tekrar çalıştırmanız gerekir.

# 1. Kayıp Eşiğini Belirleme
# Müşterilerin siparişler arasındaki ortalama süresine bakalım.
# Bu veri setinde tekrar eden müşteri az olduğu için, kayıp eşiğini genellikle 90 veya 180 gün olarak seçebiliriz.
# Olist'in uzun teslimat sürelerini ve tek seferlik alım oranını göz önüne alarak 90 günlük bir eşik belirleyelim.
CHURN_EŞİĞİ = 90  # Gün

# 2. Her müşterinin son sipariş tarihini bulma
last_purchase_date = df_master.groupby('customer_unique_id')['order_purchase_timestamp'].max().reset_index()
last_purchase_date.columns = ['customer_unique_id', 'Last_Purchase_Date']

# 3. Analiz Referans Tarihini tekrar belirleme (RFM'deki aynı tarih)
snapshot_date = df_master['order_purchase_timestamp'].max() + timedelta(days=1)

# 4. Yenilik (Recency) değerini hesaplama
last_purchase_date['Recency'] = (snapshot_date - last_purchase_date['Last_Purchase_Date']).dt.days

# 5. Hedef Değişken (Churn) Oluşturma
# Eğer Recency (Yenilik) değeri 90 günden büyükse, müşteri "Kayıp" (1) kabul edilir.
last_purchase_date['Churn'] = last_purchase_date['Recency'].apply(lambda x: 1 if x > CHURN_EŞİĞİ else 0)

print(f"Kayıp Eşiği: {CHURN_EŞİĞİ} gün")
print(f"Kayıp Olan Müşteri Sayısı: {last_purchase_date['Churn'].sum()}")
print(f"Kayıp Oranı: {last_purchase_date['Churn'].mean() * 100:.2f}%")

# rfm_results_corrected tablosuna Churn bilgisini ekleme
rfm_model_df = pd.merge(rfm_results_corrected, last_purchase_date[['customer_unique_id', 'Churn']], 
                        on='customer_unique_id', how='left')

print("\nChurn hedef değişkeni RFM tablosuna eklendi.")
print(rfm_model_df[['customer_unique_id', 'Recency', 'Churn']].head())


# Bağımsız Değişkenler (X)
features = ['Recency', 'Frequency', 'Monetary', 'K_Cluster']
X = rfm_model_df[features]

# Bağımlı Değişken (Y)
Y = rfm_model_df['Churn']

# Kategorik değişken (K_Cluster) için One-Hot Encoding yapalım (Modelin anlayacağı hale getirme)
X = pd.get_dummies(X, columns=['K_Cluster'], drop_first=True)

# Veriyi Eğitim ve Test setlerine ayırma
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42, stratify=Y)

print(f"Eğitim Seti Boyutu: {X_train.shape}")
print(f"Test Seti Boyutu: {X_test.shape}")


# Modeli başlatma ve eğitme
model = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced')
model.fit(X_train, Y_train)

# Test seti üzerinde tahmin yapma
Y_pred = model.predict(X_test)


# Modelin performansını raporlama
print("\n--- Churn Prediction Model Değerlendirmesi ---")
print(classification_report(Y_test, Y_pred))

# Katsayıları (Özellik Önemleri) Görme
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print("\n--- Özelliklerin Churn Üzerindeki Etkisi (Katsayılar) ---")
print(feature_importance)