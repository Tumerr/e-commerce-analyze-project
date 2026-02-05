import pandas as pd
from sqlalchemy import create_engine
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Veritabanı dosya yolu
DATABASE_FILE = 'C:\\Users\\tumer\\OneDrive\\Masaüstü\\e-commerce-project\\olist.db'

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
    df = pd.read_sql_query(query, engine)

    # Tarih dönüşümleri ve Yenilik (Recency) hesabı
    df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])
    snapshot_date = df['last_purchase_date'].max() + dt.timedelta(days=1)
    df['recency'] = (snapshot_date - df['last_purchase_date']).dt.days

    # Hedef Değişken (Churn): 180 günden fazladır alışveriş yapmayanlar
    df['churn'] = (df['recency'] > 180).astype(int)

    # 1. ÖZELLİK İSİMLERİNİ TÜRKÇELEŞTİRME
    # Modelin kullanacağı sütunları Türkçeye çeviriyoruz
    turkce_kolonlar = {
        'frequency': 'Alışveriş Sıklığı',
        'monetary': 'Toplam Harcama Tutarı',
        'avg_order_value': 'Ortalama Sipariş Değeri',
        'num_unique_categories': 'Alınan Farklı Kategori Sayısı',
        'avg_review_score': 'Ortalama Yorum Puanı'
    }
    df = df.rename(columns=turkce_kolonlar)
    
    # Özellik listesini yeni isimlerle tanımlıyoruz
    ozellikler = list(turkce_kolonlar.values())
    X = df[ozellikler].fillna(0)
    y = df['churn']

    # Eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Veri Ölçeklendirme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model Eğitimi (Sınıf ağırlıklarını dengeleyerek)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    # Tahminler
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("\n--- Müşteri Kaybı (Churn) Tahmin Modeli Sonuçları ---")
    report = classification_report(y_test, y_pred, output_dict=True)
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred))

    # -----------------------------------------------------
    # GRAFİK 1: Özellik Önem Düzeyleri (Türkçeleştirilmiş)
    # -----------------------------------------------------
    print("\nÖzellik Önem Düzeyleri Hesaplanıyor...")
    onem_duzeyleri = pd.Series(model.feature_importances_, index=ozellikler).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=onem_duzeyleri.values, y=onem_duzeyleri.index, hue=onem_duzeyleri.index, palette='cubehelix', legend=False)
    plt.title('Müşteri Kaybı Tahmininde Kritik Faktörler', fontsize=16)
    plt.xlabel('Etki Katsayısı (Önem Düzeyi)', fontsize=12)
    plt.ylabel('Müşteri Özellikleri', fontsize=12)
    
    output_filename = 'churn_ozellik_onemi_tr.png'
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.show()
    print(f"Grafik başarıyla '{output_filename}' adıyla kaydedildi.")
    plt.close()
    
    # -----------------------------------------------------
    # GRAFİK 2: Tahmin Dağılımı (Türkçeleştirilmiş)
    # -----------------------------------------------------
    churn_proba_df = pd.DataFrame({'Churn Olasılığı': y_pred_proba, 'Gerçek Durum': y_test})
    churn_proba_df['Gerçek Durum'] = churn_proba_df['Gerçek Durum'].map({0: 'Müşteri Kalıcı', 1: 'Müşteri Kayıp'})
    
    plt.figure(figsize=(10, 6))
    sns.histplot(churn_proba_df, x='Churn Olasılığı', hue='Gerçek Durum', 
                 kde=True, palette={'Müşteri Kalıcı': 'green', 'Müşteri Kayıp': 'red'}, 
                 bins=30, line_kws={'linewidth': 3})
    
    plt.axvline(0.5, color='gray', linestyle='--', linewidth=1.5, label='Karar Eşiği (0.5)')
    
    plt.title('Modelin Müşteri Kaybı Olasılığı Tahmin Dağılımı', fontsize=16)
    plt.xlabel('Hesaplanan Kayıp Olasılığı', fontsize=12)
    plt.ylabel('Müşteri Sayısı', fontsize=12)
    plt.legend(title='Gerçek Müşteri Durumu')
    
    output_filename_dist = 'churn_tahmin_dagilimi_tr.png'
    plt.tight_layout()
    plt.savefig(output_filename_dist)
    plt.show()
    print(f"Grafik başarıyla '{output_filename_dist}' adıyla kaydedildi.")
    plt.close()

    # -----------------------------------------------------
    # GRAFİK 3: Model Başarı Metrikleri (Türkçeleştirilmiş)
    # -----------------------------------------------------
    basari_metrikleri = {
        'Kesinlik (Precision)': report['1']['precision'],
        'Duyarlılık (Recall)': report['1']['recall'],
        'F1-Skoru': report['1']['f1-score']
    }
    
    isimler = list(basari_metrikleri.keys())
    degerler = list(basari_metrikleri.values())
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=isimler, y=degerler, hue=isimler, 
                palette=['#1f77b4', '#ff7f0e', '#2ca02c'], legend=False)
    
    for i, v in enumerate(degerler):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=12, fontweight='bold')
        
    plt.ylim(0, 1.1)
    plt.title('Model Performans Göstergeleri (Churn Sınıfı)', fontsize=16)
    plt.xlabel('Performans Ölçütü', fontsize=12)
    plt.ylabel('Puan (0-1 Arası)', fontsize=12)
    
    output_filename_metrics = 'model_basari_metrikleri_tr.png'
    plt.tight_layout()
    plt.savefig(output_filename_metrics)
    plt.show()
    print(f"Grafik başarıyla '{output_filename_metrics}' adıyla kaydedildi.")

if __name__ == '__main__':
    db_engine = get_db_engine()
    if db_engine:
        predict_churn(db_engine)
        db_engine.dispose()
        print("\nAnaliz ve görselleştirme başarıyla tamamlandı. Bağlantı kapatıldı.")