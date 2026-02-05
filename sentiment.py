import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob # Basit duygu analizi için

# 1. Veri Yükleme
path = r'C:\Users\tumer\OneDrive\Masaüstü\e-ticaret\data\\'
reviews = pd.read_csv(path + 'olist_order_reviews_dataset.csv')

# Sadece yorum metni olan satırları seçelim
df_reviews = reviews[['review_score', 'review_comment_message']].dropna()

# 2. Basit Metin Temizleme (Küçük harf yapma)
df_reviews['cleaned_text'] = df_reviews['review_comment_message'].str.lower()

print(f"Analiz edilecek yorum sayısı: {len(df_reviews)}")


def label_sentiment(score):
    if score > 3: return 'Pozitif'
    elif score == 3: return 'Nötr'
    else: return 'Negatif'

df_reviews['Sentiment'] = df_reviews['review_score'].apply(label_sentiment)

# Yorum uzunluğunu hesaplama (Karakter sayısı)
df_reviews['Review_Length'] = df_reviews['review_comment_message'].apply(len)

print(df_reviews.groupby('Sentiment')['review_score'].count())


from wordcloud import WordCloud

# Sadece negatif yorumlardaki kelimeleri birleştirelim
negatif_yorumlar = " ".join(text for text in df_reviews[df_reviews['Sentiment']=='Negatif'].cleaned_text)

# Portekizce stop words (etkisiz kelimeler) - basit bir liste
stop_words_pt = ["o", "a", "e", "que", "do", "da", "em", "um", "para", "com", "na", "no", "mais"]

wordcloud = WordCloud(width=800, height=400, 
                      background_color='white', 
                      stopwords=stop_words_pt).generate(negatif_yorumlar)

plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Negatif Yorumlarda En Çok Geçen Kelimeler (Portekizce)", fontsize=20)
plt.show()




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re







import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# --- 1. Sadece Duygu Bildiren Sözlüklerin Hazırlanması ---
# Portekizce metinden Türkçeye nokta atışı çeviri
pos_lexicon = {
    'bom': 'İyi Ürün', 'otimo': 'Harika', 'excelente': 'Mükemmel', 
    'perfeito': 'Kusursuz', 'rapido': 'Hızlı Teslimat', 'recomendo': 'Öneririm',
    'lindo': 'Güzel/Estetik', 'maravilhoso': 'Muhteşem', 'satisfeito': 'Memnuniyet',
    'original': 'Orijinal Ürün', 'legal': 'Hoş/Güzel'
}

neg_lexicon = {
    'atraso': 'Gecikme', 'defeito': 'Kusurlu/Arızalı', 'ruim': 'Kötü Kalite',
    'errado': 'Yanlış Ürün', 'pessimo': 'Berbat', 'faltando': 'Eksik Ürün',
    'quebrado': 'Kırık/Hasarlı', 'caro': 'Pahalı', 'demora': 'Yavaşlık',
    'insatisfeito': 'Memnuniyetsiz', 'falsificado': 'Sahte/Replika'
}

def get_filtered_counts(sentiment_type, lexicon):
    # Belirli duyguya ait metinleri birleştir
    text = " ".join(df_reviews[df_reviews['Sentiment'] == sentiment_type].cleaned_text)
    words = re.findall(r'\w+', text)
    # Sadece sözlükte olanları filtrele
    filtered_list = [lexicon[word] for word in words if word in lexicon]
    return pd.DataFrame(Counter(filtered_list).most_common(10), columns=['Nitelik', 'Adet'])

# --- 2. Grafik 1: Müşterileri Memnun Eden Unsurlar (Pozitif) ---
pos_df = get_filtered_counts('Pozitif', pos_lexicon)

plt.figure(figsize=(12, 7))
sns.barplot(x='Adet', y='Nitelik', data=pos_df, palette='summer')
plt.title('Müşteri Memnuniyetinin Temel Nedenleri (Olumlu Nitelikler)', fontsize=16, pad=20)
plt.xlabel('Bahsedilme Sayısı', fontsize=12)
plt.ylabel('Kritik Başarı Faktörü', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# --- 3. Grafik 2: Müşteri Kaybına Yol Açan Sorunlar (Negatif) ---
neg_df = get_filtered_counts('Negatif', neg_lexicon)

plt.figure(figsize=(12, 7))
sns.barplot(x='Adet', y='Nitelik', data=neg_df, palette='Reds_r')
plt.title('Müşteri Şikayetlerinin Kök Nedenleri (Olumsuz Nitelikler)', fontsize=16, pad=20)
plt.xlabel('Bahsedilme Sayısı', fontsize=12)
plt.ylabel('Kritik Sorun Alanı', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

