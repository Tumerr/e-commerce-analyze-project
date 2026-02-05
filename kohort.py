import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import pandas as pd

# Veri yollarını tanımlayın
path = r'C:\Users\tumer\OneDrive\Masaüstü\e-ticaret\data\\'

# Temel tabloları yükleyin
orders = pd.read_csv(path + 'olist_orders_dataset.csv')
items = pd.read_csv(path + 'olist_order_items_dataset.csv')
customers = pd.read_csv(path + 'olist_customers_dataset.csv')

# Tabloları birleştirerek df_master oluşturun
df_master = pd.merge(orders, items, on='order_id')
df_master = pd.merge(df_master, customers, on='customer_id')

# Tarih sütununu datetime formatına çevirin (Kohort analizi için kritik)
df_master['order_purchase_timestamp'] = pd.to_datetime(df_master['order_purchase_timestamp'])

print("df_master başarıyla oluşturuldu. Satır sayısı:", len(df_master))
# 1. Sipariş Ayını Belirleme
df_master['order_month'] = df_master['order_purchase_timestamp'].dt.to_period('M')

# 2. Her Müşterinin İlk Sipariş Ayını (Kohort) Bulma
df_master['cohort_month'] = df_master.groupby('customer_unique_id')['order_purchase_timestamp'] \
    .transform('min').dt.to_period('M')

# 3. Kohort İndeksi Hesaplama (İlk siparişten bu yana kaç ay geçti?)
def get_date_int(df, column):
    year = df[column].dt.year
    month = df[column].dt.month
    return year, month

order_year, order_month = get_date_int(df_master, 'order_month')
cohort_year, cohort_month = get_date_int(df_master, 'cohort_month')

years_diff = order_year - cohort_year
months_diff = order_month - cohort_month

# +1 ekliyoruz çünkü ilk ay '1. ay' olarak başlasın
df_master['cohort_index'] = years_diff * 12 + months_diff + 1

# 4. Aktif Müşterileri Sayma
cohort_data = df_master.groupby(['cohort_month', 'cohort_index'])['customer_unique_id'].nunique().reset_index()

# 5. Pivot Tablo Oluşturma
cohort_pivot = cohort_data.pivot(index='cohort_month', columns='cohort_index', values='customer_unique_id')

# 6. Retention Rate (Tutma Oranı) Hesaplama (%)
cohort_sizes = cohort_pivot.iloc[:, 0]
retention = cohort_pivot.divide(cohort_sizes, axis=0)

plt.figure(figsize=(20, 10))
sns.heatmap(retention, annot=True, fmt='.1%', cmap='YlGnBu', vmin=0.0, vmax=0.05)
plt.title('Müşteri Tutma Oranı (Retention Rate) - Kohort Analizi', fontsize=18)
plt.xlabel('Aylar (İlk Siparişten Sonra)', fontsize=12)
plt.ylabel('Kohort Ayı (İlk Sipariş)', fontsize=12)
plt.show()