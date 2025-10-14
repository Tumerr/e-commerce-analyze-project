import pandas as pd
from sqlalchemy import create_engine
import datetime as dt

DATABASE_FILE = 'olist.db'

def export_rfm_data():
    
    try:
        connection_str = f'sqlite:///{DATABASE_FILE}'
        engine = create_engine(connection_str)
        print(f"'{DATABASE_FILE}' veritabanına bağlanılıyor...")
    except Exception as e:
        print(f"HATA: Veritabanına bağlanılamadı. Hata detayı: {e}")
        return

    query = """
    SELECT
        c.customer_unique_id,
        o.order_id,
        o.order_purchase_timestamp,
        op.payment_value
    FROM olist_orders o
    JOIN olist_customers c ON o.customer_id = c.customer_id
    JOIN olist_order_payments op ON o.order_id = op.order_id
    WHERE o.order_status = 'delivered';
    """
    print("RFM analizi için veri çekiliyor...")
    df = pd.read_sql_query(query, engine)

    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    snapshot_date = df['order_purchase_timestamp'].max() + dt.timedelta(days=1)

    rfm_data = df.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': lambda date: (snapshot_date - date.max()).days,
        'order_id': 'nunique',
        'payment_value': 'sum'
    }).rename(columns={'order_purchase_timestamp': 'Recency', 'order_id': 'Frequency', 'payment_value': 'Monetary'})

    rfm_data['R_Score'] = pd.qcut(rfm_data['Recency'], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm_data['F_Score'] = pd.qcut(rfm_data['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm_data['M_Score'] = pd.qcut(rfm_data['Monetary'], 5, labels=[1, 2, 3, 4, 5]).astype(int)

    def assign_segment(row):
        if row['R_Score'] >= 4 and row['F_Score'] >= 4: return 'Şampiyonlar'
        if row['R_Score'] >= 4 and row['F_Score'] >= 2: return 'Sadık Müşteriler'
        if row['R_Score'] >= 3 and row['F_Score'] >= 3: return 'Potansiyel Sadıklar'
        if row['R_Score'] >= 4 and row['F_Score'] == 1: return 'Yeni Müşteriler'
        if row['R_Score'] >= 3 and row['F_Score'] < 2: return 'İlgilenilmesi Gerekenler'
        if row['R_Score'] < 3 and row['F_Score'] >= 3: return 'Uyumaya Başlayan Sadıklar'
        if row['R_Score'] < 3 and row['F_Score'] < 3: return 'Risk Altındaki Müşteriler'
        return 'Diğer'

    rfm_data['Segment'] = rfm_data.apply(assign_segment, axis=1)


    output_csv_file = 'rfm_results.csv'
    rfm_data.to_csv(output_csv_file, index=True)
    print(f"RFM sonuçları başarıyla '{output_csv_file}' dosyasına kaydedildi.")
    engine.dispose()

if __name__ == '__main__':
    export_rfm_data()