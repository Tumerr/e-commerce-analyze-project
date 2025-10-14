import pandas as pd
from sqlalchemy import create_engine
import seaborn as sns
import matplotlib.pyplot as plt


DATABASE_FILE = 'olist.db'


def get_db_engine():
    
    try:
        
        connection_str = f'sqlite:///{DATABASE_FILE}'
        engine = create_engine(connection_str)
        print(f"'{DATABASE_FILE}' veritabanına bağlanılıyor...")
        return engine
    except Exception as e:
        print(f"HATA: Veritabanına bağlanılamadı. Hata detayı: {e}")
        return None

def analyze_top_product_categories(engine):
  
    
    if engine is None:
        return

    print("\n--- Analiz 1: En Çok Gelir Getiren Ürün Kategorileri ---")
    
    
    query = """
    SELECT
        t.product_category_name_english AS category,
        SUM(oi.price) AS total_revenue
    FROM
        olist_order_items AS oi
    JOIN
        olist_products AS p ON oi.product_id = p.product_id
    JOIN
        product_category_name_translation AS t ON p.product_category_name = t.product_category_name
    JOIN
        olist_orders AS o ON oi.order_id = o.order_id
    WHERE
        o.order_status = 'delivered'
        AND p.product_category_name IS NOT NULL
    GROUP BY
        category
    ORDER BY
        total_revenue DESC
    LIMIT 10;
    """
    
    print("Sorgu 1 çalıştırılıyor...")

    df_top_categories = pd.read_sql_query(query, engine)
        
    print("\nEn Çok Gelir Getiren İlk 10 Kategori:")
    print(df_top_categories)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='total_revenue', y='category', data=df_top_categories,hue='category',legend=False)
    plt.title('En Çok Gelir Getiren İlk 10 Ürün Kategorisi', fontsize=16)
    plt.xlabel('Toplam Gelir (Revenue)', fontsize=12)
    plt.ylabel('Ürün Kategorisi', fontsize=12)
    plt.show()

def analyze_monthly_revenue(engine):
    
    if engine is None:return
    print("\n---Analiz 2: Aylık Gelir Trendleri---")


    query = """
    SELECT 
        strftime('%Y-%m', o.order_purchase_timestamp) AS month,
        SUM(oi.price) AS monthly_revenue
    FROM
        olist_orders AS o
    JOIN
        olist_order_items AS oi ON o.order_id = oi.order_id
    WHERE
        o.order_status = 'delivered'
    GROUP BY 
        month
    ORDER BY
        month;
    """

    print("Sorgu 2 çalıştırılıyor...")
    df_monthly_revenue = pd.read_sql_query(query,engine)

    df_monthly_revenue = df_monthly_revenue.iloc[:-1,:]

    print("\nAylık Toplam Gelir Verisi:")
    print(df_monthly_revenue)

    plt.figure(figsize=(15,7))
    sns.lineplot(x='month',y='monthly_revenue',data=df_monthly_revenue, marker='o', color= 'purple')
    plt.title('Aylık Toplam Gelir Trendi', fontsize=16)
    plt.xlabel('Ay', fontsize=12)
    plt.ylabel('Toplam Gelir(Revenue)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    db_engine = get_db_engine()
    
    if db_engine:
        analyze_top_product_categories(db_engine)
        analyze_monthly_revenue(db_engine)  
        db_engine.dispose()
        print("\nİşlem tamamlandı. Bağlantı kapatıldı.")