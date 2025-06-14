
###################################################
# PROJE: Rating Product & Sorting Reviews in Amazon
###################################################

###################################################
# İş Problemi
###################################################

# E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış sonrası verilen puanların doğru şekilde hesaplanmasıdır.
# Bu problemin çözümü e-ticaret sitesi için daha fazla müşteri memnuniyeti sağlamak, satıcılar için ürünün öne çıkması ve satın
# alanlar için sorunsuz bir alışveriş deneyimi demektir. Bir diğer problem ise ürünlere verilen yorumların doğru bir şekilde sıralanması
# olarak karşımıza çıkmaktadır. Yanıltıcı yorumların öne çıkması ürünün satışını doğrudan etkileyeceğinden dolayı hem maddi kayıp
# hem de müşteri kaybına neden olacaktır. Bu 2 temel problemin çözümünde e-ticaret sitesi ve satıcılar satışlarını arttırırken müşteriler
# ise satın alma yolculuğunu sorunsuz olarak tamamlayacaktır.

###################################################
# Veri Seti Hikayesi
###################################################

# Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile çeşitli metadataları içermektedir.
# Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı puanları ve yorumları vardır.

# Değişkenler:
# reviewerID: Kullanıcı ID’si
# asin: Ürün ID’si
# reviewerName: Kullanıcı Adı
# helpful: Faydalı değerlendirme derecesi
# reviewText: Değerlendirme
# overall: Ürün rating’i
# summary: Değerlendirme özeti
# unixReviewTime: Değerlendirme zamanı
# reviewTime: Değerlendirme zamanı Raw
# day_diff: Değerlendirmeden itibaren geçen gün sayısı
# helpful_yes: Değerlendirmenin faydalı bulunma sayısı
# total_vote: Değerlendirmeye verilen oy sayısı



###################################################
# GÖREV 1: Average Rating'i Güncel Yorumlara Göre Hesaplayınız ve Var Olan Average Rating ile Kıyaslayınız.
###################################################
import pandas as pd
import numpy as np
from scipy.stats import norm

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır.
# Bu görevde amacımız verilen puanları tarihe göre ağırlıklandırarak değerlendirmek.
# İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.


###################################################
# Adım 1: Veri Setini Okutunuz ve Ürünün Ortalama Puanını Hesaplayınız.
###################################################
amazon_df = pd.read_csv("/Users/sametgirgin/PycharmProjects/Cases/Rating Product&SortingReviewsinAmazon/amazon_review.csv")
amazon_df.head(20)
amazon_df.shape

average_ratings = amazon_df['overall'].mean()
#4.587589013224822
###################################################
# Adım 2: Tarihe Göre Ağırlıklı Puan Ortalamasını Hesaplayınız.
###################################################
def time_based_weighted_average(dataframe, w1=0.28, w2=0.26, w3=0.24, w4=0.22):
    return (dataframe[dataframe["day_diff"] <= 30]["overall"].mean() * w1 +
            dataframe[(dataframe["day_diff"] > 30) & (dataframe["day_diff"] <= 90)]["overall"].mean() * w2 +
            dataframe[(dataframe["day_diff"] > 90) & (dataframe["day_diff"] <= 180)]["overall"].mean() * w3 +
            dataframe[dataframe["day_diff"] > 180]["overall"].mean() * w4)

# Ağırlıklı ortalamayı hesaplayalım
weighted_average = time_based_weighted_average(amazon_df)
# 4.6987161061560725
###################################################
# Görev 2: Ürün için Ürün Detay Sayfasında Görüntülenecek 20 Review'i Belirleyiniz.
###################################################


###################################################
# Adım 1. helpful_no Değişkenini Üretiniz
###################################################

# Not:
# total_vote bir yoruma verilen toplam up-down sayısıdır.
# up, helpful demektir.
# veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.
amazon_df['helpful_no'] = amazon_df['total_vote'] - amazon_df['helpful_yes']
amazon_df.head(10)

###################################################
# Adım 2. score_pos_neg_diff, score_average_rating ve wilson_lower_bound Skorlarını Hesaplayıp Veriye Ekleyiniz
###################################################
# score_pos_neg_diff: up - down
amazon_df['score_pos_neg_diff'] = amazon_df['helpful_yes'] - amazon_df['helpful_no']

# score_average_rating: up / (up + down)
amazon_df['score_average_rating'] = amazon_df.apply(
    lambda x: 0 if (x['helpful_yes'] + x['helpful_no']) == 0
    else x['helpful_yes'] / (x['helpful_yes'] + x['helpful_no']),
    axis=1)

# wilson_lower_bound: En güvenilir puanlama skoru
def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = norm.ppf(1 - (1 - confidence) / 2)
    phat = up / n
    return (phat + z*z/(2*n) - z * np.sqrt((phat*(1 - phat) + z*z/(4*n)) / n)) / (1 + z*z/n)

amazon_df['wilson_lower_bound'] = amazon_df.apply(
    lambda x: wilson_lower_bound(x['helpful_yes'], x['helpful_no']), axis=1
)

amazon_df[['helpful_yes', 'helpful_no', 'score_pos_neg_diff', 'score_average_rating', 'wilson_lower_bound']].head(50)

##################################################
# Adım 3. 20 Yorumu Belirleyiniz ve Sonuçları Yorumlayınız.
###################################################
top_20_reviews = amazon_df.sort_values("wilson_lower_bound", ascending=False).head(20)
top_20_reviews = amazon_df.sort_values("score_average_rating", ascending=False).head(20)


#Çıkarımlar:
#En üst sıradaki yorumlar genellikle yüksek faydalı oy (helpful_yes) almış ve az sayıda faydasız (helpful_no) oy almıştır.
# WLB skoru, sadece oranı değil, aynı zamanda toplam oy sayısını da dikkate aldığından daha sağlam bir sıralama sağlar.


