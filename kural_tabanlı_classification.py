# Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama
import pandas as pd

# Bir oyun şirketi müşterilerinin bazı özelliklerini kullanarak seviye tabanlı (level based)
# yeni müşteri tanımları (persona) oluşturmak ve bu yeni müşteri tanımlarına göre segmentler
# oluşturup bu segmentlere göre yeni gelebilecek müşterilerin şirkete ortalama ne kadar
# kazandırabileceğini tahmin etmek istemektedir.

persona_df = pd.read_csv("persona.csv")

# Persona.csv veri seti uluslararası bir oyun şirketinin sattığı ürünlerin fiyatlarını ve bu 
# ürünleri satın alan kullanıcıların bazı demografik bilgilerini barındırmaktadır. Veri seti 
# her satış işleminde oluşan kayıtlardan meydana gelmektedir. Bunun anlamı tablo 
# tekilleştirilmemiştir. Diğer bir ifade ile belirli demografik özelliklere sahip bir 
# kullanıcı birden fazla alışveriş yapmış olabilir.

                                    ## Görev 1
#Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
persona_df.head()
persona_df.describe()
persona_df.info()

# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?
persona_df['SOURCE'].unique()
persona_df['SOURCE'].nunique()

#Soru 3: Kaç unique PRICE vardır?
persona_df["PRICE"].nunique()
#Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
price_counts = persona_df['PRICE'].value_counts().sort_index()
price_counts

# Soru 5: Hangi ülkeden kaçar tane satış olmuş?
country_counts = persona_df['COUNTRY'].value_counts().sort_index()
country_counts

# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?
revenue_per_country = persona_df.groupby("COUNTRY")["PRICE"].sum().sort_values(ascending=False)
revenue_per_country

#Soru 7: SOURCE türlerine göre satış sayıları nedir?
sales_per_source = persona_df["SOURCE"].value_counts()
sales_per_source

#Soru 8: Ülkelere göre PRICE ortalamaları nedir?
avg_price_per_country = persona_df.groupby("COUNTRY")["PRICE"].mean().sort_values(ascending=False)
avg_price_per_country

#Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?
avg_price_per_source = persona_df.groupby("SOURCE")["PRICE"].mean().sort_values(ascending=False)
avg_price_per_source

#Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
avg_price_country_source = persona_df.groupby(["COUNTRY", "SOURCE"])["PRICE"].mean().sort_values(ascending=False).sort_index()
avg_price_country_source

                                        ## Görev 2

