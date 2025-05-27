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
persona_df.shape[0]
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
# COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?

#avg_price_detailed = persona_df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"])["PRICE"].mean().reset_index()

pivot_df = persona_df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"])["PRICE"].mean().to_frame(name="PRICE")
pivot_df
                                        ## Görev 3
# Çıktıyı PRICE’a göre sıralayınız.

agg_df = pivot_df.sort_values(by="PRICE", ascending=False)
agg_df
                                        ## Görev 4
#Indekste yer alan isimleri değişken ismine çeviriniz.
agg_df = agg_df.reset_index()
agg_df

                                        ## Görev 5
#Age değişkenini kategorik değişkene çeviriniz ve agg_df’e ekleyiniz.
bins = [0, 18, 23, 30, 40, agg_df["AGE"].max()]
labels = ["0_18", "19_23", "24_30", "31_40", "41_70"]

agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins=bins, labels=labels, right=False)
agg_df

                                        ## Görev 6
#Yeni seviye tabanlı müşterileri (persona) tanımlayınız.

# customers_level_based sütunu oluşturma
agg_df["customers_level_based"] = [
    f"{row['COUNTRY'].upper()}_{row['SOURCE'].upper()}_{row['SEX'].upper()}_{row['AGE_CAT']}"
    for _, row in agg_df.iterrows()
]
#Bu kısım, agg_df DataFrame'ini customers_level_based sütunundaki benzersiz değerlere göre
# gruplar. as_index=False parametresi sayesinde, gruplama sonucu oluşturulan DataFrame'in
# gruplama sütunu indeks olarak kullanılmaz, bunun yerine normal bir sütun olarak kalır.

agg_df_unique = agg_df.groupby("customers_level_based", as_index=False).agg({
    "COUNTRY": "first",  # Take the first value for 'COUNTRY'
    "SOURCE": "first",   # Take the first value for 'SOURCE'
    "SEX": "first",      # Take the first value for 'SEX'
    "PRICE": "mean",     # Calculate the mean for 'PRICE'
    "AGE_CAT": "first"   # Take the first value for 'AGE_CAT'
})

#agg_df_unique = agg_df.groupby("customers_level_based", as_index=False)["PRICE"].mean().sort_values(by="PRICE", ascending=False).reset_index(drop=True)
#agg_df_unique

                                            ## Görev 7
#Yeni müşterileri (personaları) segmentlere ayırınız.
agg_df_unique["SEGMENT"] = pd.qcut(agg_df_unique["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df_unique

segment_summary = agg_df_unique.groupby("SEGMENT",observed=True).agg({"PRICE": ["mean", "max", "sum"]}).round(2)
segment_summary

                                            ## Görev 8
new_persona = "FRA_IOS_FEMALE_31_40"
#new_persona = "TUR_ANDROID_FEMALE_31_40"
agg_df_unique[agg_df_unique["customers_level_based"]== new_persona].assign(PRICE=lambda x: x["PRICE"].round(2))


#Bir fonksiyon yardımıyla
def predict_customer_segment(country, source, sex, age, agg_df_unique):
    # Yaş kategorisini belirle
    if age < 18:
        age_cat = "0_18"
    elif age < 23:
        age_cat = "19_23"
    elif age < 30:
        age_cat = "24_30"
    elif age < 40:
        age_cat = "31_40"
    else:
        age_cat = "41_70"

    # Persona'yı oluştur
    persona = f"{country.upper()}_{source.upper()}_{sex.upper()}_{age_cat}"

    # Persona'yı agg_df_unique içinde ara
    match = agg_df_unique[agg_df_unique["customers_level_based"] == persona]

    # Sonuç varsa döndür
    if not match.empty:
        segment = match["SEGMENT"].values[0]
        price = float(match["PRICE"].values[0])  # float64 yerine float dönüşümü
        return {
            "persona": persona,
            "segment": segment,
            "expected_revenue": round(price, 2)
        }
    else:
        return {
            "persona": persona,
            "segment": "Not found",
            "expected_revenue": None
        }

result = predict_customer_segment("fra", "ios", "female", 35, agg_df_unique)

# 3. metod
def gelir(agg_df_unique, country, source, sex, age_cat):
    filters = {"COUNTRY": country, "SOURCE": source, "SEX": sex, "AGE_CAT": age_cat}

    for key, value in filters.items():
        agg_df_unique = agg_df_unique[agg_df_unique[key] == value] if value else agg_df_unique
    return agg_df_unique["PRICE"].mean()

# gelir fonksiyonunu çağır
gelir(agg_df_unique, country="tur", source="android", sex="female", age_cat="31_40")


# agg_df columns kontrol et
print(agg_df_unique.columns)

#Son eklemeler yapıldı.




