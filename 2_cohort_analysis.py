import pandas as pd
from sqlalchemy import create_engine
import seaborn as sns 
import matplotlib.pyplot as plt


DATABASE_FILE = 'olist.db'


def get_db_engine():
    try:
        connection_str =f'sqlite:///{DATABASE_FILE}'
        engine = create_engine(connection_str)
        print(f"'{DATABASE_FILE}' veritabanına bağlanıyor...")
        return engine
    except Exception as e:
        print(f"HATA: Veritabanına bağlanılamadı. Hata detayı:{e}")
        return None
    
def analyze_cohort_retention(engine):

    if engine is None: return
    print("\n---Analiz 3 : Müşteri Kohort Analizi---")


    query = """
    WITH customer_first_order AS (

        SELECT
            c.customer_unique_id,
            MIN(strftime('%Y-%m', o.order_purchase_timestamp))  AS cohort_month
        FROM
            olist_orders o 
        JOIN
            olist_customers c ON o.customer_id = c.customer_id
        GROUP BY
            c.customer_unique_id
    ),

    monthly_orders AS (

        SELECT
            DISTINCT
            c.customer_unique_id,
            strftime('%Y-%m', o.order_purchase_timestamp) AS order_month
        FROM
            olist_orders o
        JOIN
            olist_customers c ON o.customer_id = c.customer_id
    )
    SELECT
        cfo.cohort_month,
        (CAST(strftime('%Y',mo.order_month || '-01') AS INTEGER) - CAST(strftime('%Y',cfo.cohort_month || '-01') AS INTEGER)) *12 + 
        (CAST(strftime('%m',mo.order_month || '-01') AS INTEGER) - CAST(strftime('%m',cfo.cohort_month || '-01') AS INTEGER)) AS month_number,
        COUNT(DISTINCT cfo.customer_unique_id) AS num_customers
    FROM
        customer_first_order cfo
    JOIN
        monthly_orders mo ON cfo.customer_unique_id = mo.customer_unique_id
    GROUP BY
        cfo.cohort_month, month_number

    """


    print("İleri Seviye SQL sorgusu çalıştırılıyor...")
    df_cohort_raw = pd.read_sql_query(query,engine)

    cohort_pivot = df_cohort_raw.pivot_table(index='cohort_month',columns='month_number',values='num_customers')

    cohort_sizes = cohort_pivot.iloc[:,0]

    retention_matrix = cohort_pivot.divide(cohort_sizes,axis=0)*100

    print("\nKohort Analizi Isı Haritası Oluşturuluyor:")

    plt.figure(figsize=(18,10))
    sns.heatmap(retention_matrix,annot=True, fmt='.1f', cmap='viridis', vmin=0, vmax=10)
    plt.title('Aylık Müşteri Elde Tutma (Retention) Oranları (%)  - Kohort Analizi', fontsize=16)
    plt.xlabel('İlk Alışverişten Sonraki Ay Sayısı', fontsize=12)
    plt.ylabel('Müşteri Kazanım Ayı (Kohort)', fontsize=12)
    plt.show()



if __name__ == '__main__':
    db_engine = get_db_engine()
    if db_engine:
        analyze_cohort_retention(db_engine)
        db_engine.dispose()
        print("\nAnaliz Tamamlandı. Bağlantı Kapatıldı.")
        