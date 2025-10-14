import pandas as pd
from sqlalchemy import create_engine
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns


DATABASE_FILE = 'olist.db'


def get_db_engine():
    try:
        connection_str = f'sqlite:///{DATABASE_FILE}'
        engine = create_engine(connection_str)
        print(f"'{DATABASE_FILE}' veritabanına bağlanılıyor...")
        return engine
    except Exception as e:
        print(f"HATA:Veritabanına bağlanılamadı. Hata detayı:{e}")
        return None
    

def perform_market_basket_analysis(engine):
    
    if engine is None: return
    print("\n---Pazar Sepeti Analizi---")

    query = """
    SELECT
        o.order_id,
        t.product_category_name_english as category
    FROM
        olist_orders o
    JOIN
        olist_order_items oi ON o.order_id = oi.order_id
    JOIN
        olist_products p ON oi.product_id = p.product_id
    JOIN
        product_category_name_translation t ON p.product_category_name = t.product_category_name
    WHERE
        o.order_status = 'delivered'
        AND t.product_category_name_english IS NOT NULL;
"""

    print("Sipariş ve kategori verileri çekiliyor...")
    df = pd.read_sql_query(query, engine)


    basket_data = df.groupby('order_id')['category'].apply(list).tolist()
    
    te = TransactionEncoder()
    te_ary = te.fit(basket_data).transform(basket_data)
    df_onehot = pd.DataFrame(te_ary, columns=te.columns_)

    print("Veri analize hazırlandı. Sık kullanılan kategori setleri bulunuyor...")

    frequent_itemsets = apriori(df_onehot, min_support=0.00001, use_colnames=True)
    
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    sorted_rules = rules.sort_values("lift", ascending=False)

    print("\nEn Güçlü Birliktelik Kuralları ( En Yüksek Lift Değerine Göre):")
    print(sorted_rules[['antecedents','consequents','support','confidence','lift']].head(20))

    
    print("\nBirliktelik kuralları görselleştiriliyor...")

    top_10_rules = sorted_rules.head(10)
    
    top_10_rules['antecedents_str'] = top_10_rules['antecedents'].apply(lambda x: ','.join(list(x)))
    top_10_rules['consequents_str'] = top_10_rules['consequents'].apply(lambda x: ','.join(list(x)))
    top_10_rules['rule_text'] = top_10_rules.apply(lambda row: f"{row['antecedents_str']} -> {row['consequents_str']}", axis=1)

    plt.figure(figsize=(12,8))
    ax = sns.barplot(x="lift", y="rule_text", data=top_10_rules, palette="crest")

    ax.bar_label(ax.containers[0], fmt='%.1f', padding=5, fontsize=12 )

    plt.title('En Güçlü 10 Ürün Kategorisi İlişkisi', fontsize=16)
    plt.xlabel('Lift Değeri (İlişki Gücü)', fontsize=12)
    plt.ylabel('Birliktelik Kuralı', fontsize=12)
    plt.xlim(0,top_10_rules['lift'].max() * 1.1)
    plt.tight_layout()

    output_filename = 'market_basket_top_rules.png'
    plt.savefig(output_filename)
    print(f"\nGrafik başarıyla '{output_filename}' adıyla kaydedildi.")
    plt.show()


if __name__ == '__main__':
    db_engine = get_db_engine()
    if db_engine:
        perform_market_basket_analysis(db_engine)
        db_engine.dispose()
        print("\nAnaliz tamamlandı. Bağlantı kapatıldı.")