import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
import seaborn as sns
# --- 1. VERİ YÜKLEME VE BİRLEŞTİRME ---
path = r'C:\Users\tumer\OneDrive\Masaüstü\e-ticaret\data\\' 

orders = pd.read_csv(path + 'olist_orders_dataset.csv')
order_items = pd.read_csv(path + 'olist_order_items_dataset.csv')
products = pd.read_csv(path + 'olist_products_dataset.csv')
category_translation = pd.read_csv(path + 'product_category_name_translation.csv')

# Tabloları birleştir (Sipariş ID ve Kategori İsmi lazım)
df = pd.merge(order_items, products, on='product_id', how='left')
df = pd.merge(df, category_translation, on='product_category_name', how='left')

# İngilizce kategori ismini al, yoksa orijinali kalsın
df['category'] = df['product_category_name_english'].fillna(df['product_category_name'])

# --- 2. SEPET MATRİSİ (BASKET) OLUŞTURMA ---
# Her satır bir sipariş (order_id), her sütun bir kategori olacak şekilde veriyi pivot edelim
basket = (df.groupby(['order_id', 'category'])['order_item_id']
          .count().unstack().reset_index().fillna(0)
          .set_index('order_id'))

# Değerleri Binary (0 veya 1) yapalım: 1'den büyük olanlar (aynı kategoriden birden fazla ürün) 1 olsun
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)

# Sadece 1'den fazla farklı kategori içeren sepetleri filtreleyelim (Birliktelik bulmak için)
basket_filter = basket_sets[(basket_sets.sum(axis=1) > 1)]

print(f"Analiz edilecek çoklu ürün içeren sepet sayısı: {len(basket_filter)}")

# --- 3. SIK ÜRÜN KÜMELERİNİ BULMA (SUPPORT) ---
# min_support: Bir ürün grubunun tüm sepetlerin en az yüzde kaçında birlikte görülmesi gerektiğini belirler
frequent_itemsets = apriori(basket_filter, min_support=0.01, use_colnames=True)

# --- 4. KURALLARI OLUŞTURMA (CONFIDENCE & LIFT) ---
# En az 0.1 (yüzde 10) güvene sahip kuralları getirelim
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

# Sonuçları Lift değerine göre sıralayalım (Lift > 1 olması anlamlı bir ilişkiyi gösterir)
rules = rules.sort_values('lift', ascending=False)

print("\n--- En Güçlü Birliktelik Kuralları ---")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

tr_kategori_map = {
    'home_confort': 'Ev Konforu',
    'bed_bath_table': 'Yatak Banyo Sofra',
    'furniture_decor': 'Mobilya Dekorasyon',
    'construction_tools_lights': 'İnşaat Aletleri Aydınlatma',
    'home_construction': 'Ev İnşaat',
    'perfumery': 'Parfümeri',
    'health_beauty': 'Sağlık Güzellik',
    'toys': 'Oyuncak',
    'baby': 'Bebek',
    'cool_stuff': 'Havalı Ürünler',
    'housewares': 'Züccaciye',
    'garden_tools': 'Bahçe Gereçleri'
}

# --- GÜNCELLENEN KISIM: ETİKETLERİ TÜRKÇELEŞTİRME ---
# Kategorileri Türkçe sözlüğe göre çeviriyoruz, sözlükte yoksa orijinali kalıyor.
rules['antecedents_tr'] = rules['antecedents'].apply(lambda x: ', '.join([tr_kategori_map.get(i, i) for i in list(x)]))
rules['consequents_tr'] = rules['consequents'].apply(lambda x: ', '.join([tr_kategori_map.get(i, i) for i in list(x)]))

# Kural ismini Türkçe okla oluşturma
rules['rule_name_tr'] = rules['antecedents_tr'] + ' -> ' + rules['consequents_tr']

# En yüksek Confidence değerine sahip ilk 10 kuralı alalım
top_conf_rules = rules.sort_values('confidence', ascending=False).head(10)

plt.figure(figsize=(12, 8))
# Y ekseninde artık Türkçe isimlerin olduğu 'rule_name_tr' kullanılıyor
sns.barplot(x='confidence', y='rule_name_tr', data=top_conf_rules, palette='Reds_r')

# Çubukların üzerine yüzde değerlerini yazalım
for index, value in enumerate(top_conf_rules['confidence']):
    plt.text(value, index, f' %{value*100:.1f}', va='center', fontsize=12, fontweight='bold')

plt.title('En Yüksek Satın Alma İhtimaline (Güven) Sahip Ürün İkilileri', fontsize=16)
plt.xlabel('Güven Oranı (Confidence)', fontsize=12)
plt.ylabel('Ürün İlişkisi (Eğer A Alınırsa -> B Alınır)', fontsize=12)
plt.xlim(0, 1.1) 
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()