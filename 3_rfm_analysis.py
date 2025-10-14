import pandas as pd
from sqlalchemy import create_engine
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns


DATABASE_FILE ='olist.db'


def get_db_engine():

    try:
        connection_str = f'sqlite:///{DATABASE_FILE}'
        engine = create_engine(connection_str)
        print(f"'{DATABASE_FILE}' veri tabanına baplanıyor...")
        return engine
    except Exception as e:
        print(f"HATA: Veri Tabanına Bağlanılamadı. Hata detayı:{e}")
        return None

def perform_rfm_analysis(engine):
    if engine is None : return


    query = """
    SELECT 
        c.customer_unique_id,
        o.order_id,
        o.order_purchase_timestamp,
        op.payment_value
    FROM
        olist_orders o
    JOIN
        olist_customers c ON o.customer_id = c.customer_id
    JOIN
        olist_order_payments op ON o.order_id = op.order_id
    WHERE
        o.order_status = 'delivered';
"""    

    print("RFM analizi için veri çekiliyor...")
    df= pd.read_sql_query(query,engine)


    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

    snapshot_date = df['order_purchase_timestamp'].max() + dt.timedelta(days=1)
    print(f"Analiz tarihi (Snapshot Date):{snapshot_date.date()}")


    rfm_data = df.groupby('customer_unique_id').agg({
        'order_purchase_timestamp':lambda date: (snapshot_date-date.max()).days,
        'order_id':'nunique',
        'payment_value':'sum'
     }).rename(columns={'order_purchase_timestamp':'Recency','order_id':'Frequency','payment_value':'Monetary'})
    

    rfm_data['R_Score'] = pd.qcut(rfm_data['Recency'],5, labels=[5,4,3,2,1])
    rfm_data['F_Score'] = pd.qcut(rfm_data['Frequency'].rank(method='first'),5,labels=[1,2,3,4,5])
    rfm_data['M_Score'] = pd.qcut(rfm_data['Monetary'],5, labels=[1,2,3,4,5])


    def assign_segment(row):
        if row['R_Score'] >= 4 and row['F_Score'] >= 4 :
            return 'Şampiyonlar'
        if row['R_Score'] >= 4 and row['F_Score'] >= 2 :
            return 'Sadık Müşteriler'
        if row['R_Score'] >= 3 and row['F_Score'] >= 3 :
            return 'Potansiyel Sadıklar'
        if row['R_Score'] >= 4 and row['F_Score'] == 1 :
            return 'Yeni Müşteriler'
        if row['R_Score'] >= 3 and row['F_Score'] < 2 :
            return 'İlgilenilmesi Gerekenler'
        if row['R_Score'] < 3 and row['F_Score'] >= 3 :
            return 'Uyumaya Başlayan Sadıklar'
        if row['R_Score'] < 3 and row['F_Score'] < 3 : 
            return 'Risk Altındaki Müşteriler' 
        return 'Diğer'

    rfm_data['Segment'] = rfm_data.apply(assign_segment, axis=1)

    print("\nRFM Analiz Tamamlandı. Segmentlere Göre Müşteri Dağılımı:")
    segment_counts =rfm_data['Segment'].value_counts()
    print(segment_counts)


    plt.figure(figsize=(15,8))

    ax = sns.barplot(x=segment_counts.index, y=segment_counts.values , palette='rocket')

    for p in ax.patches:
        ax.annotate(format(p.get_height(),'.0f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va= 'center',
                    xytext= (0,9),
                    textcoords= 'offset points',
                    fontsize=12)    
    plt.title('Müşteri Segmentlerine Göre Dağılım', fontsize=16)
    plt.xlabel('Segment',fontsize=12)
    plt.ylabel('Müşteri Sayısı',fontsize=12) 
    plt.xticks(rotation=45, ha="right")
    
    ax.set_ylim(0,segment_counts.max()* 1.1)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__' :
    db_engine = get_db_engine()
    if db_engine:
        perform_rfm_analysis(db_engine)
        db_engine.dispose()
        print("\nAnaliz tamamlandı. Bağlantı kapatıldı.")

    
    
    
    
    
    
    
