import pandas as pd
import numpy as np

# Güncel dosya yolu
path = r'C:\Users\tumer\OneDrive\Masaüstü\e-ticaret\data\\' 

# 9 farklı CSV dosyasını okuma
customers = pd.read_csv(path + 'olist_customers_dataset.csv')
orders = pd.read_csv(path + 'olist_orders_dataset.csv')
order_items = pd.read_csv(path + 'olist_order_items_dataset.csv')
products = pd.read_csv(path + 'olist_products_dataset.csv')
reviews = pd.read_csv(path + 'olist_order_reviews_dataset.csv')
payments = pd.read_csv(path + 'olist_order_payments_dataset.csv')
category_translation = pd.read_csv(path + 'product_category_name_translation.csv')

print("Tüm Veri Setleri Başarıyla Yüklendi.")

df_master = pd.merge(orders, customers, on='customer_id', how='left')

# 2. Ürün Detayları (order_id üzerinden)
df_master = pd.merge(df_master, order_items, on='order_id', how='left')

# 3. Ürün Bilgileri (product_id üzerinden)
df_master = pd.merge(df_master, products, on='product_id', how='left')

# 4. Kategori İsimlerini İngilizceye Çevirme
# İngilizce çeviri tablosunu birleştirme
df_master = pd.merge(df_master, category_translation, 
                     on='product_category_name', 
                     how='left')

# İngilizce kategoriyi ana sütun yapma
df_master.rename(columns={'product_category_name_english': 'product_category'}, 
                 inplace=True)
df_master.drop('product_category_name', axis=1, inplace=True) # Portekizce sütunu düşür

print(f"Ana Tablo (df_master) Başlangıç Boyutu: {df_master.shape}")

datetime_cols = [col for col in df_master.columns if 'timestamp' in col or 'date' in col]

for col in datetime_cols:
    df_master[col] = pd.to_datetime(df_master[col], errors='coerce')

    # 1. Eksik Değerleri Yönetme: 
# product_id veya customer_unique_id eksik olan satırlar atılır.
df_master.dropna(subset=['product_id', 'customer_unique_id'], inplace=True)

# 2. İptal Edilmiş/Kullanışsız Siparişleri Filtreleme:
# RFM için sadece 'delivered' (teslim edilmiş) siparişleri kullanmak en güvenli yoldur.
df_master = df_master[df_master['order_status'] == 'delivered'].copy() 

# 3. Harcama Tutarını Hesaplama: 
# Bir ürün için toplam harcama = item price + freight value (Kargo ücreti)
df_master['Total_Price'] = df_master['price'] + df_master['freight_value']

# 4. RFM için gerekli anahtar müşteri ID'si
rfm_df = df_master[['customer_unique_id', 'order_purchase_timestamp', 'Total_Price']].copy()

print(f"\nTemizlik ve Filtreleme Sonrası Ana Tablo Boyutu: {rfm_df.shape}")
print(f"Benzersiz Teslim Edilmiş Müşteri Sayısı: {rfm_df['customer_unique_id'].nunique()}")