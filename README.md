[📊 Olist E-Ticaret Veri Analizi Pro.md](https://github.com/user-attachments/files/23494115/Olist.E-Ticaret.Veri.Analizi.Pro.md)
# 📊 Olist E-Ticaret Veri Analizi Projesi  
*Müşteri Davranışları, Sadakat ve Büyüme Fırsatları*

Bu proje, büyük ölçekli e-ticaret verilerini işleyerek müşteri davranışlarını anlamlandırmak, stratejik kararları desteklemek ve operasyonel verimliliği artırmak amacıyla geliştirilmiştir. Çalışmada betimsel analizler, denetimli ve denetimsiz makine öğrenmesi algoritmaları ve doğal dil işleme yöntemleri kullanılmıştır,.


📂 Veri Seti Hakkında

Projede, Eylül 2016 ile Ekim 2018 tarihleri arasındaki yaklaşık 100.000 siparişi içeren kamuya açık bir e-ticaret veri seti kullanılmıştır. Veriler ilişkisel bir yapıya sahip olup 8 ana tablodan oluşmaktadır ve SQL sorguları ile birleştirilerek analize hazırlanmıştır:

<img width="788" height="474" alt="image" src="https://github.com/user-attachments/assets/84f0596e-7837-46ce-afac-8a8a39612171" />

• Müşteri ve Konum: olist_customers_dataset, olist_geolocation_dataset
• Sipariş ve Ürün: olist_orders_dataset, olist_order_items_dataset, olist_products_dataset
• Ödeme ve Değerlendirme: olist_order_payments_dataset, olist_order_reviews_dataset
• Satıcı: olist_sellers_dataset
🛠️ Kullanılan Teknolojiler ve Yöntemler
• Programlama Dilleri: Python, SQL.
• Analiz Yaklaşımları: Kohort Analizi, RFM Segmentasyonu, Churn Tahmini, Pazar Sepeti Analizi, Duygu Analizi.

--------------------------------------------------------------------------------
📊 Analiz Süreci ve Modeller
Proje kapsamında ham verinin bilgiye dönüştürülmesi için aşağıdaki 5 temel analiz adımı uygulanmıştır:

1. Müşteri Sadakati: Kohort Analizi (Cohort Analysis)

Müşterilerin ilk alışveriş yaptıkları aya göre gruplandırılarak zaman içindeki davranışları incelenmiştir.
<img width="989" height="392" alt="image" src="https://github.com/user-attachments/assets/c468a08b-6e94-4b55-b1f2-7996ac37996d" />

• Amaç: Müşteri elde tutma (retention) oranlarını belirlemek.
• Bulgu: İlk aydan sonra müşteri elde tutma oranlarının oldukça düşük olduğu ve platformdaki alışverişlerin büyük çoğunluğunun tek seferlik olduğu tespit edilmiştir.


2. Müşteri Segmentasyonu: RFM Analizi ve K-Means

Müşteriler; Recency (Güncellik), Frequency (Sıklık) ve Monetary (Parasal Değer) metriklerine göre puanlanmış ve K-Means kümeleme algoritması ile segmentlere ayrılmıştır.
<img width="893" height="470" alt="image" src="https://github.com/user-attachments/assets/27f5acfb-2c41-428a-9274-7fbdb0538e56" />

• Segmentler: Potansiyel Müşteriler, Büyük Harcamacılar, Risk Altındakiler, Sadıklar.
• Bulgu: En büyük kitleyi "Yeni/Düşük Harcama Yapanlar" oluştururken, "Sadık ve Yüksek Harcama Yapanlar" en küçük gruptur,.


3. Müşteri Kaybı Tahmini (Churn Prediction)

Müşterilerin platformu terk etme olasılıklarını tahmin etmek için Random Forest algoritması kullanılmıştır.
<img width="764" height="553" alt="image" src="https://github.com/user-attachments/assets/f5e3311f-3820-4178-a691-634fbc4cb37a" />
<img width="628" height="484" alt="image" src="https://github.com/user-attachments/assets/29930359-b40d-470c-9f4c-fc312f4370e7" />
<img width="681" height="522" alt="image" src="https://github.com/user-attachments/assets/345bc7d9-1a57-42ab-a9cb-4f549e8f5372" />


• Yöntem: Son alışveriş tarihine göre 0-180 gün ve 180-360 gün aralıkları için modeller eğitilmiştir. Sınıf dengesizliği için ağırlıklandırma yapılmıştır,.
• Kritik Özellikler: Toplam harcama tutarı, ortalama sipariş değeri ve yorum puanı.
• Performans: 0-180 gün aralığı için modelin doğruluğu (Accuracy) %69.30, duyarlılığı (Recall) %75.00 olarak ölçülmüştür.


4. Birliktelik Kuralları: Market Sepet Analizi

Ürünler arasındaki ilişkileri keşfetmek ve çapraz satış fırsatlarını yakalamak için Apriori Algoritması kullanılmıştır.
<img width="863" height="518" alt="image" src="https://github.com/user-attachments/assets/095668ca-7445-4a69-9d50-976290a32bd8" />

• Metrikler: Destek (Support), Güven (Confidence), Kaldıraç (Lift).
• Bulgu: "Ev Konforu" kategorisinden ürün alanların %86 ihtimalle "Yatak Banyo Sofra" kategorisinden de ürün aldığı görülmüştür.


5. Müşteri Geri Bildirimi: Duygu Analizi (Sentiment Analysis)

Yaklaşık 40.000 müşteri yorumu, doğal dil işleme yöntemi olan BERT tabanlı model ile analiz edilmiştir.
<img width="685" height="480" alt="image" src="https://github.com/user-attachments/assets/35738347-c997-4b78-b629-4f71206d0acc" />
<img width="762" height="461" alt="image" src="https://github.com/user-attachments/assets/1bb05c7e-09ee-424c-86fa-66236b5c4c9d" />
<img width="695" height="529" alt="image" src="https://github.com/user-attachments/assets/d6569f7f-7f99-477e-a225-8bd434b49fe1" />

• Pozitif Nedenler: "İyi ürün", "Hızlı teslimat", "Öneririm".
• Negatif Nedenler: "Yanlış ürün", "Gecikme", "Kusurlu ürün".
• Tutarlılık: 5 puan veren müşterilerin yorumlarının %75'i model tarafından pozitif olarak etiketlenmiştir, bu da puan-yorum tutarlılığının yüksek olduğunu göstermektedir,.

--------------------------------------------------------------------------------


##Veriye Dayalı Stratejik Öneriler

Yapılan tüm analizler sonucunda, şirketin büyümesini sürdürülebilir kılmak ve karlılığını artırmak için 
aşağıdaki aksiyonların alınması önerilmektedir:

1. Müşteri Elde Tutma Stratejileri Geliştirilmeli: Kohort analizi, müşteri sadakatinin kritik derecede düşük olduğunu göstermiştir. 
Özellikle "Risk Altındaki" ve "Uyumaya Başlayan" RFM segmentlerine yönelik acil "geri kazanma" kampanyaları 
(örneğin, kişiselleştirilmiş indirimler, "sizi özledik" e-postaları) hayata geçirilmelidir.

2. Operasyonel Süreçler İyileştirilmeli: Duygu analizi, müşteri şikayetlerinin temelinde geciken teslimatlar ve 
hatalı/bozuk ürünler olduğunu kanıtlamıştır. Lojistik ve kalite kontrol süreçlerinin gözden geçirilmesi, müşteri memnuniyetini ve 
dolayısıyla sadakati artırmak için en öncelikli adımdır.

3. Çapraz Satış (Cross-Sell) Fırsatları Değerlendirilmeli: Pazar sepeti analizi, yatak/banyo/masa ile market_place gibi güçlü ürün 
birliktelikleri ortaya çıkarmıştır. Bu ilişkiler, ürün öneri motorlarında ve "paket teklif" kampanyalarında aktif olarak kullanılmalıdır.

4. Tahmin Modeli Aktif Olarak Kullanılmalı: Geliştirilen churn modeli, riskli müşterileri yüksek başarıyla tespit etmektedir. 
Bu modelin çıktıları, pazarlama otomasyon sistemlerine entegre edilerek, churn riski en yüksek müşterilere ayrılmadan önce 
proaktif olarak özel teklifler sunulmalı ve müşteri hizmetleri tarafından özel ilgi gösterilmelidir.


