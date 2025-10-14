import pandas as pd
from sqlalchemy import create_engine
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



DATABASE_FILE = 'olist.db'


def get_db_engine():
    try:
        connection_str = f'sqlite:///{DATABASE_FILE}'
        engine = create_engine(connection_str)
        print(f"'{DATABASE_FILE}' veri tabanına bağlanılıyor...")
        return engine
    except Exception as e:
        print(f"HATA: veritabanına bağlanılamadı. Hata detayı:{e}")
        return None
    
def predict_churn(engine):

    if engine is None: return

    query = """
    SELECT 
        c.customer_unique_id,
        COUNT(DISTINCT O.order_id) AS frequency,
        SUM(op.payment_value) AS monetary,
        MIN(o.order_purchase_timestamp) AS first_purchase_date,
        MAX(o.order_purchase_timestamp) AS last_purchase_date,
        AVG(op.payment_value) AS avg_order_value,
        COUNT(DISTINCT p.product_category_name) AS num_unique_categories,
        AVG(r.review_score) AS avg_review_score
    FROM
        olist_customers c
    JOIN
        olist_orders o ON c.customer_id = o.customer_id
    JOIN
        olist_order_payments op ON o.order_id = op.order_id
    LEFT JOIN
        olist_order_items oi ON o.order_id = oi.order_id
    LEFT JOIN
        olist_products p ON oi.product_id = p.product_id
    LEFT JOIN
        olist_order_reviews r ON o.order_id = r.order_id
    WHERE
        o.order_status = 'delivered'
    GROUP BY
        c.customer_unique_id;
"""


    print("Churn analizi için özellik seti oluşturuluyor...")
    df = pd.read_sql_query(query,engine)

    df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])
    snapshot_date = df['last_purchase_date'].max() + dt.timedelta(days=1)
    df['recency'] = (snapshot_date - df['last_purchase_date']).dt.days

    df['churn'] = (df['recency'] > 180).astype(int)

    features = ['frequency', 'monetary', 'avg_order_value', 'num_unique_categories', 'avg_review_score']
    X = df[features].fillna(0)
    y = df['churn']


    X_train_, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    print("\n---Churn Tahmin Modeli Sonuçları---")
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test,y_pred))


    print("\nÖzellik Önem Düzeyleri:")
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print(importances)

    plt.figure(figsize=(10,6))
    sns.barplot(x=importances.values, y=importances.index, hue=importances.index, palette='cubehelix', legend=False)
    plt.title('Churn Tahmininde Özelliklerin Önem Düzeyi', fontsize=16)
    plt.xlabel('Önem Düzeyi', fontsize=12)
    plt.ylabel('Özellikler', fontsize=12)

    output_filename = 'churn_feature_importance.png'
    plt.savefig(output_filename)
    plt.tight_layout()
    print(f"\nGrafik başarıyla '{output_filename}' adıyla kaydedildi.")
    plt.show()


if __name__ == '__main__' :
    db_engine = get_db_engine()
    if db_engine:
        predict_churn(db_engine)
        db_engine.dispose()
        print("\nAnaliz tamamlandı. Bağlantı kapatıldı.")