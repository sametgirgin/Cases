##################################################
# Pandas Alıştırmalar
##################################################

import numpy as np
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#########################################
# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
#########################################
titanic = sns.load_dataset("titanic")
titanic.head()
#########################################
# Görev 2: Yukarıda tanımlanan Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
#########################################
gender = titanic['sex'].value_counts()
gender
#########################################
# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.
#########################################
unique_values = titanic.nunique()
unique_values
#########################################
# Görev 4: pclass değişkeninin unique değerleri bulunuz.
#########################################
unique_pclass = titanic['pclass'].unique()
unique_pclass #passenger class
#########################################
# Görev 5:  pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
#########################################
unique_counts = titanic[['pclass', 'parch']].nunique()
unique_counts #parent + children

#########################################
# Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz.
# Tekrar tipini kontrol ediniz.
#########################################
print(titanic["embarked"].dtype)

titanic['embarked'] = titanic['embarked'].astype('category')

print(titanic["embarked"].dtype)

#########################################
# Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.
#########################################
c_embarked = titanic[titanic['embarked'] == 'C']
c_embarked.head()
#########################################
# Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.
#########################################
non_s_embarked = titanic[titanic['embarked'] != 'S']
non_s_embarked.head()
#########################################
# Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
#########################################
young_female = titanic[(titanic['age'] < 30) & (titanic['sex'] == 'female')]
young_female.head()
#########################################
# Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
#########################################
titanic[(titanic['fare'] > 500) | (titanic['age'] > 70)]

#########################################
# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
#########################################

titanic.isnull().sum()

#########################################
# Görev 12: who değişkenini dataframe'den düşürün.
#########################################
titanic.drop(columns=['who'])

#########################################
# Görev 13: deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode)
# ile doldurunuz.
#########################################

mode_value = titanic['deck'].mode()[0]
titanic['deck'] = titanic['deck'].fillna(mode_value)
print(titanic['deck'].isnull().sum()) # check null values --> 0

#########################################
# Görev 14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurun.
#########################################

age_median = titanic['age'].median()

titanic['age'] = titanic['age'].fillna(age_median)

print(titanic['age'].isnull().sum())

#########################################
# Görev 15: survived değişkeninin Pclass ve Cinsiyet değişkenleri kırılımınında sum,
# count, mean değerlerini bulunuz.
#########################################
agg_funcs = ['sum', 'count', 'mean']
results = titanic.groupby(['pclass', 'sex'])['survived'].agg(agg_funcs)

print(results)

#########################################
# Görev 16:  30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon
# yazınız.Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken
# oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)
#########################################
titanic['age_flag'] = titanic['age'].apply(lambda age: 1 if age < 30 else 0)

print(titanic[['age', 'age_flag']].head(10))

#########################################
# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
#########################################

tips = sns.load_dataset("tips")

print(tips.head())

#########################################
# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill
# değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################

results = tips.groupby('time', observed="True")['total_bill'].agg(['sum', 'min', 'max', 'mean'])

print(results)

# observed=False: Varsayılan davranış — tüm kategori kombinasyonlarını gösterir (kullanılmasa bile).
#observed=True: Yalnızca veri içinde gerçekten var olan kombinasyonları gösterir.

#########################################
# Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını
# bulunuz.
#########################################
agg_funcs_time = ['sum', 'min', 'max', 'mean']

results_time_day = tips.groupby(['day', 'time'], observed=True)['total_bill'].agg(agg_funcs_time)
print(results_time_day)

#########################################
# Görev 20:Lunch zamanına ve kadın müşterilere ait total_bill ve tip  değerlerinin
# day'e göre toplamını, min, max ve ortalamasını bulunuz.
#########################################

filtered_lunch_female = tips[(tips['time'] == 'Lunch') & (tips['sex'] == 'Female')]

results_lunch_female = filtered_lunch_female.groupby('day')[['total_bill', 'tip']].agg(['sum', 'min', 'max', 'mean'])

print(results_lunch_female)

#########################################
# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir?
#########################################
#filtered_size_totalbill = tips[(tips['size'] < 3) & (tips['total_bill'] > 10)]
filtered_size_totalbill = tips.loc[(tips['size'] < 3) & (tips['total_bill'] > 10)]

avg_values = filtered_size_totalbill.mean(numeric_only=True)
print(avg_values)

#########################################
# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturun. Her bir müşterinin ödediği
# totalbill ve tip in toplamını versin.
#########################################

tips['total_bill_tip_sum'] = tips['total_bill'] + tips['tip']
tips.head()


#########################################
# Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız
# ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
#########################################
top_30 = tips.sort_values(by='total_bill_tip_sum', ascending=False).head(30)
top_30


