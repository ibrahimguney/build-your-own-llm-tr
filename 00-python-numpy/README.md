# Modül 00 — Python ve NumPy Temelleri

> "Bir matrisin satırı mı yoksa sütunu mu derken kafan karışıyorsa, daha attention'a geçemezsin."

Bu modül, sonraki tüm modüllerin **dilbilgisi**dir. LLM'in matematiği, derin öğrenme kütüphaneleri, eğitim döngüleri — hepsi NumPy/PyTorch tarzı **vektörel düşünmeyi** kabul eder. Burada o düşünce tarzının zemini atılıyor.

---

## 🎯 Bu Modülde Ne Öğreneceksin?

1. **NumPy `ndarray`**'i bir veri konteynerinden ibaret görmeyi bırakacak, onu "şekli olan bir tensör" olarak düşünmeye başlayacaksın.
2. **Broadcasting**'i içselleştireceksin — `for` döngüsü yazma refleksin kırılacak.
3. **Eksen (axis)** kavramını hatasız kullanabileceksin: "axis=0 satırlar boyunca mı sütunlar boyunca mı?" sorusu artık tereddüt yaratmayacak.
4. **Rastgele sayı üretimi** ile ağırlık başlangıcı (weight init), örnekleme ve tekrarlanabilirlik arasındaki bağı kurabileceksin.
5. **Türev sezgisi**ni sayısal olarak hissedeceksin — Modül 02'de göreceğin `autograd`'ın "neden gerekli" olduğunu kavramış olarak gireceksin.

---

## 🧭 Bu Modülün Yol Haritası

| Dosya | Kavram | Neden Burada? |
|-------|--------|---------------|
| `01_python_hatirlatma.py` | Python kısa tekrar | Sonraki dosyalarda kullanılacak Python idiyomlarını (list comprehension, `*args`, basit class) ortak zemine çekmek için. |
| `02_numpy_array_temelleri.py` | `ndarray` doğuşu, indeksleme | NumPy'nin temel veri yapısını tanımıyorsak hiçbir şey öğrenemeyiz. |
| `03_broadcasting.py` | Boyutların hizalanması | LLM'lerde batch boyutuyla işlem yaparken hayat kurtaran kural. |
| `04_vektor_matris_islemleri.py` | Element-wise vs `matmul` | `*` ile `@` farkı, attention'da `Q @ K.T` çarpımının kalbi. |
| `05_eksenler_ve_reshape.py` | Şekli yeniden düzenlemek | Transformer'da sürekli `(B, T, C)` ↔ `(B*T, C)` dönüşümleri yapılır. |
| `06_rastgelelik.py` | Stokastik dünya | Ağırlık init, dropout, sampling — hepsi rastgelelikle başlar. |
| `07_turev_sezgisi.py` | Eğimi sayıyla görmek | Gradyan inişini ezbere değil sezgiyle anlamak için. |
| `notebook.ipynb` | İnteraktif keşif | Yukarıdakileri grafiklerle ve canlı denemelerle pekiştirir. |

---

## 📐 Ön Bilgi

- Temel Python (değişken, fonksiyon, döngü) bilgisi yeterli.
- Matris ve vektör kavramına aşinalık iyi olur ama şart değil; **Modül 01** lineer cebiri detaylı işliyor — bu modül onun zeminini hazırlıyor.

---

## ▶️ Nasıl Çalıştırılır?

Her `.py` dosyası bağımsızdır:

```bash
python 02_numpy_array_temelleri.py
```

Dosyanın altındaki `if __name__ == "__main__":` bloğu içindeki `assert`'ler geçerse, kod doğru çalışıyor demektir.

Notebook için:

```bash
jupyter lab notebook.ipynb
```

---

## ✅ Modül Sonu Kontrol Listesi

Bu modülü bitirdiğinde aşağıdakileri **döngü kullanmadan** yapabilmelisin:

- [ ] Bir matrisin her sütununu o sütunun ortalamasıyla normalize etmek.
- [ ] İki vektör arasındaki **kosinüs benzerliği**ni tek satırda hesaplamak.
- [ ] `(B, T, C)` şeklinde bir tensörü `(B*T, C)`'ye dönüştürüp geri almak.
- [ ] `f(x) = x^2` fonksiyonunun `x=3` noktasındaki türevini sayısal yaklaşıkla bulmak.
- [ ] `np.random.default_rng(seed)` ile tekrarlanabilir bir deney kurmak.

Bunları zorlanmadan yapabiliyorsan, **Modül 01 — Lineer Cebir**'e geçebilirsin.

---

## 🔗 Sonraki Adım

[`01-lineer-cebir/`](../01-lineer-cebir/) — Matris çarpımının geometrik anlamı, özdeğer/özvektörler, SVD ile boyut indirgeme.
