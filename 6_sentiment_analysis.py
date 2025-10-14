import pandas as pd
from sqlalchemy import create_engine
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re


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
    
def perform_sentiment_analysis(engine):
    if engine is None:return
    print("\n---Duygu Analizi (Sentiment Analysis)---")

    query = """
    SELECT
        review_score,
        review_comment_message
    FROM
        olist_order_reviews
    WHERE
        review_comment_message IS NOT NULL;
"""

    print("Müşteri yorumları veritabanına çekiliyor...")
    df_reviews = pd.read_sql_query(query,engine)

    print("Hugging Face'den duygu analizi yükleniyor...")
    sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    print("Model başarıyla yüklendi.")

    sample_df = df_reviews.head(500).copy()
    print(f"/nAnaliz {len(sample_df)} adet yorum üzerinde yapılacak...")


    sample_df['sentiment_result'] = sample_df['review_comment_message'].apply(lambda x: sentiment_pipeline(x[:512]))
    sample_df['sentiment_label'] = sample_df['sentiment_result'].apply(lambda x: x[0]['label'])


    def map_sentiment(label):
        stars = int(re.search(r'\d+',label).group())
        if stars >= 4: return 'Positive'
        if stars <= 2: return 'Negative'
        return 'Neutral'
    
    sample_df['sentiment_category'] = sample_df['sentiment_label'].apply(map_sentiment)

    print("\nModelin Duygu Tahminleri ile Gerçek Müşteri Puanlarının Karşılaştırılması:")
    cross_tab = pd.crosstab(sample_df['review_score'], sample_df['sentiment_category'], margins=True)
    print(cross_tab)

    print("\nOlumsuz yorumlardan kelime bulutu oluşturuluyor...")
    negative_comments_df = sample_df[sample_df['sentiment_category'] == 'Negative']


    if not negative_comments_df.empty:
        text = " ".join(comment for comment in negative_comments_df['review_comment_message'])
        
        stopwords_pt = ['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'com', 'não', 'uma', 'os', 'no', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'foi', 'ao', 'ele', 'das', 'tem', 'à', 'seu', 'sua', 'ou', 'seja', 'qual', 'quando', 'muito', 'só', 'já', 'está', 'eu', 'também', 'meu', 'pela', 'pelo', 'sem', 'depois', 'mesmo', 'produto', 'pedido', 'pra', 'foi', 'me']
        wordcloud = WordCloud(width=1200, height=600, background_color="white", stopwords=stopwords_pt).generate(text)

        plt.figure(figsize=(15,7))  
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title("Olumsuz Yorumlarda En Sık Geçen Kelimeler", fontsize=16)

        output_filename = 'negative_comment_wordcloud.png'
        plt.savefig(output_filename)
        print(f"\nKelime bulutu grafiği başarıyla '{output_filename}' adıyla kaydedildi.")
        plt.show()


    else:
        print("Analiz edilen örneklemde olumsuz yorum bulunmadı.")


if __name__ == '__main__':
    db_engine = get_db_engine()
    if db_engine:
        perform_sentiment_analysis(db_engine)
        db_engine.dispose()
        print("\nAnaliz tamamlandı. Bağlantı kapatıldı.")
