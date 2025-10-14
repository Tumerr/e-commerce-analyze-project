
import pandas as pd
from sqlalchemy import create_engine
import folium


DATABASE_FILE = 'olist.db'
GEOJSON_FILE = 'brazil-states.geojson'



def get_db_engine():
    try:
        connection_str = f'sqlite:///{DATABASE_FILE}'
        engine = create_engine(connection_str)
        print(f"'{DATABASE_FILE}' veritabanına bağlanılıyor...")
        return engine
    except Exception as e:
        print(f"HATA: Veritabanına bağlanılamadı. Hata detayı: {e}")
        return None
    
def perform_geo_analysis_sql(engine):
    if engine is None:return None
    
    print("\n---Coğrafi Analizler---")

    query_revenue = """
    SELECT 
        c.customer_state,
        SUM(op.payment_value) as total_revenue
    FROM
        olist_orders o
    JOIN
        olist_customers c ON o.customer_id = c.customer_id
    JOIN
        olist_order_payments op ON o.order_id = op.order_id
    WHERE
        o.order_status = 'delivered'
    GROUP BY
        c.customer_state
    ORDER BY
        total_revenue DESC;
"""

    query_customers = """
    SELECT
        customer_state,
        COUNT(DISTINCT customer_unique_id) as total_customers
    FROM
        olist_customers
    GROUP BY
        customer_state
    ORDER BY
        total_customers DESC;
"""


    print("\nEyaletlere Göre Toplam Gelir:")
    df_revenue = pd.read_sql_query(query_revenue,engine)
    print(df_revenue.head(10))

    print("\nEyaletlere Göre Müşteri Sayısı:")
    df_customers = pd.read_sql_query(query_customers,engine)
    print(df_customers.head(10))

    return df_revenue

def create_brazil_sales_map(df_revenue_by_state):
    if df_revenue_by_state is None: return

    print("\nİnteraktif satış haritası oluşturuluyor...")

    brazil_center = [-14.2350,-51.9253]

    m = folium.Map(location=brazil_center, zoom_start=4)

    folium.Choropleth(
        geo_data=GEOJSON_FILE,
        name='choroplet',
        data=df_revenue_by_state,
        columns=['customer_state', 'total_revenue'],
        key_on='feature.properties.sigla',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Toplam Gelir (Revenue)'
    ).add_to(m)

    output_filename = 'brazil_sales_map.html'
    m.save(output_filename)
    print(f"\nHarita başarıyla '{output_filename}' adıyla kaydedildi.")
    print("Bu dosyayı web tarayıcınızda açarak interaktif haritayı görebilirsiniz.")



if __name__ == '__main__':
    db_engine = get_db_engine()
    if db_engine:
        df_revenue = perform_geo_analysis_sql(db_engine)

        create_brazil_sales_map(df_revenue)

        db_engine.dispose()
        print("\nAnaliz tamamlandı. Bağlantı kapatıldı.")