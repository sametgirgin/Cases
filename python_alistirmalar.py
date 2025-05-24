###############################################
# Python Alıştırmalar
###############################################

###############################################
# GÖREV 1: Veri yapılarının tipleriniz inceleyiniz.
###############################################

x = 8

y = 3.2

z = 8j + 18

a = "Hello World"


b = True


c = 23 < 22



l = [1, 2, 3, 4,"String",3.2, False]



d = {"Name": "Jake",
     "Age": [27,56],
     "Adress": "Downtown"}


t = ("Machine Learning", "Data Science")



s = {"Python", "Machine Learning", "Data Science","Python"}

print(type(x))  # <class 'int'>
print(type(y))  # <class 'int'>
print(type(z))  # <class 'int'>
print(type(a))  # <class 'int'>
print(type(b))  # <class 'int'>
print(type(c))  # <class 'int'>
print(type(l))  # <class 'int'>
print(type(d))  # <class 'int'>
print(type(t))  # <class 'int'>
print(type(s))  # <class 'int'>


###############################################
# GÖREV 2: Verilen string ifadenin tüm harflerini büyük harfe çeviriniz. Virgül ve nokta yerine space koyunuz, kelime kelime ayırınız.
###############################################

text = "The goal is to turn data into information, and information into insight."

# Tüm harfleri büyük harfe çevir
text_upper = text.upper()

# Virgül ve noktaları boşlukla değiştir
text_cleaned = text_upper.replace(",", " ").replace(".", " ")

# Kelime kelime ayır
words = text_cleaned.split()

print(words)

###############################################
# GÖREV 3: Verilen liste için aşağıdaki görevleri yapınız.
###############################################

lst = ["D","A","T","A","S","C","I","E","N","C","E"]
# Adım 1: Verilen listenin eleman sayısına bakın.
print(len(lst))
# Adım 2: Sıfırıncı ve onuncu index'teki elemanları çağırın.
print(lst[0], lst[10])
# Adım 3: Verilen liste üzerinden ["D","A","T","A"] listesi oluşturun.
print(lst[0:4])
# Adım 4: Sekizinci index'teki elemanı silin.
del lst[8]
# Adım 5: Yeni bir eleman ekleyin.
lst.append("N")
# Adım 6: Sekizinci index'e  "N" elemanını tekrar ekleyin.
lst.insert(8, "N")

print(lst)
###############################################
# GÖREV 4: Verilen sözlük yapısına aşağıdaki adımları uygulayınız.
###############################################
dict = {'Christian': ["America",18],
        'Daisy':["England",12],
        'Antonio':["Spain",22],
        'Dante':["Italy",25]}

# Adım 1: Key değerlerine erişiniz.
dict.keys()
# Adım 2: Value'lara erişiniz.
dict.values()
# Adım 3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.
dict["Daisy"][1] = 13
# Adım 4: Key değeri Ahmet value değeri [Turkey,24] olan yeni bir değer ekleyiniz.
dict["Ahmet"] = ["Turkey",24]
# Adım 5: Antonio'yu dictionary'den siliniz.
dict.pop("Antonio")
print(dict)
###############################################
# GÖREV 5: Arguman olarak bir liste alan, listenin içerisindeki tek ve çift sayıları ayrı listelere atıyan ve bu listeleri return eden fonskiyon yazınız.
###############################################
l = [2,13,18,93,22]
def even_odd(lst):
    even = []
    odd = []
    for i in lst:
        if i % 2 == 0:
            even.append(i)
        else:
            odd.append(i)
    return even, odd

even, odd = even_odd(l)

print("Tek sayılar:", odd)
print("Çift sayılar:", even)
###############################################
# GÖREV 6: Aşağıda verilen listede mühendislik ve tıp fakülterinde dereceye giren öğrencilerin isimleri bulunmaktadır.
# Sırasıyla ilk üç öğrenci mühendislik fakültesinin başarı sırasını temsil ederken son üç öğrenci de tıp fakültesi öğrenci sırasına aittir.
# Enumarate kullanarak öğrenci derecelerini fakülte özelinde yazdırınız.
###############################################

ogrenciler = ["Ali","Veli","Ayşe","Talat","Zeynep","Ece"]

# Mühendislik Fakültesi ve Tıp Fakültesi öğrencilerini ayıralım
muhendislik_ogrencileri = ogrenciler[:3]
tip_ogrencileri = ogrenciler[3:]

# Mühendislik Fakültesi için Derece 1-3 arası yazdıralım
for index, ogrenci in enumerate(muhendislik_ogrencileri, 1):
    print(f"Mühendislik Fakültesi {index} . öğrenci: {ogrenci}")

# Tıp Fakültesi için Derece 1-3 arası yazdıralım
for index, ogrenci in enumerate(tip_ogrencileri, 1):
    print(f"Tıp Fakültesi {index} . öğrenci: {ogrenci}")


###############################################
# GÖREV 7: Aşağıda 3 adet liste verilmiştir. Listelerde sırası ile bir dersin kodu, kredisi ve kontenjan bilgileri yer almaktadır. Zip kullanarak ders bilgilerini bastırınız.
###############################################

ders_kodu = ["CMP1005","PSY1001","HUK1005","SEN2204"]
kredi = [3,4,2,4]
kontenjan = [30,75,150,25]

for kod, krd, kon in (zip(ders_kodu, kredi, kontenjan)):
    print(f"Kredisi {krd} olan {kod} kodlu dersin kontenjanı {kon} kişidir")

###############################################
# GÖREV 8: Aşağıda 2 adet set verilmiştir.
# Sizden istenilen eğer 1. küme 2. kümeyi kapsiyor ise ortak elemanlarını eğer kapsamıyor ise 2. kümenin 1. kümeden farkını yazdıracak fonksiyonu tanımlamanız beklenmektedir.
###############################################

kume1 = set(["data", "python"])
kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])
def ortak_elemanlar(kume1, kume2):
    if kume1.issuperset(kume2):
        ortak_elemanlar = kume1.intersection(kume2)
        print("Kapsıyor. Ortak elemanlar:" , ortak_elemanlar)
    else:
        fark = kume1.symmetric_difference(kume2)
        print("Kapsamıyor. 2. kümenin 1. kümeden farkı:", fark)

ortak_elemanlar(kume1, kume2)