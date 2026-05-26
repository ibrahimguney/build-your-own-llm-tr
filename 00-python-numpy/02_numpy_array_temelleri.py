"""
02 — NumPy Array Temelleri

NumPy `ndarray`, projenin geri kalanında çalışacağımız temel veri yapısının atasıdır.
PyTorch `Tensor` da büyük ölçüde aynı arayüzü taklit ettiği için burada öğrenilen her
şey doğrudan transfer edilebilir.

Bu dosyada şu soruların cevaplarını arıyoruz:
  - Bir array'i nasıl yaratırız? Neden farklı yollar var?
  - `dtype`, `shape`, `ndim` ne anlatır?
  - İndeksleme/dilimleme Python listesinden ne farkı var?
  - Fancy ve boolean indexing ne işe yarar?
"""

import numpy as np


# -------------------------------------------------------------
# 1) Array oluşturma — neden bu kadar çok yol var?
# -------------------------------------------------------------

def array_olusturma() -> None:
    # En düz yol: Python listesinden çevir.
    # Küçük, elle yazılmış örneklerde idealdir.
    a = np.array([1, 2, 3, 4])

    # arange: 0..N-1 aralığı. Range gibi ama doğrudan ndarray döner.
    # Eğitim örneklerini indekslerken sıkça kullanacağız.
    b = np.arange(5)

    # linspace: iki uç dahil eşit aralıklı N nokta.
    # Grafik çizerken x ekseni üretmek için ideal — `arange` ile karıştırma:
    # arange adım miktarını, linspace nokta sayısını alır.
    c = np.linspace(0.0, 1.0, 5)

    # zeros/ones: belirtilen şekilde sıfır/bir dolu tensör.
    # Ağırlıkları sıfırla başlatmak kötü bir fikirdir (Modül 06'da görülecek),
    # ama bias init için sıkça kullanılır.
    sifirlar = np.zeros((2, 3))
    birler = np.ones((2, 3))

    assert a.shape == (4,)
    assert b.tolist() == [0, 1, 2, 3, 4]
    assert np.isclose(c[-1], 1.0)
    assert sifirlar.sum() == 0
    assert birler.sum() == 6


# -------------------------------------------------------------
# 2) shape, ndim, dtype — bir tensörün "kimlik kartı"
# -------------------------------------------------------------

def kimlik_kartı() -> None:
    # 2x3 bir matris kuralım.
    m = np.array([[1, 2, 3],
                  [4, 5, 6]])

    # shape: her eksendeki uzunluk.
    # ndim: kaç eksen olduğu (yani shape'in uzunluğu).
    # size: toplam eleman sayısı.
    assert m.shape == (2, 3)
    assert m.ndim == 2
    assert m.size == 6

    # dtype, elemanların tipi. Varsayılan int64 (platforma göre değişebilir).
    # Bellek ve hesap maliyeti açısından kritik: LLM eğitiminde sıkça float32/float16 kullanırız.
    assert m.dtype in (np.int64, np.int32)

    # Tipi explicit yazmak alışkanlık edinilmeli — özellikle çarpımlar
    # int taşması (overflow) veya float dönüşüm sürpriziyle bizi vurmasın.
    f = np.array([1, 2, 3], dtype=np.float32)
    assert f.dtype == np.float32


# -------------------------------------------------------------
# 3) İndeksleme ve dilimleme
# -------------------------------------------------------------

def indeksleme() -> None:
    # 1D'de Python listesine çok benziyor.
    x = np.arange(10)
    assert x[0] == 0
    assert x[-1] == 9
    assert x[2:5].tolist() == [2, 3, 4]
    assert x[::2].tolist() == [0, 2, 4, 6, 8]

    # 2D'de tek bir köşeli parantez içinde virgülle eksenleri ayırmak,
    # Python listelerine göre asıl konfor kaynağıdır.
    # m[i, j] ↔ "i. satır, j. sütun".
    m = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

    assert m[0, 0] == 1
    assert m[2, 2] == 9

    # Satır olarak bir dilim almak:
    assert m[1].tolist() == [4, 5, 6]
    # Sütun olarak bir dilim almak — bu Python listesinde tek satırda yapılamaz.
    assert m[:, 1].tolist() == [2, 5, 8]


# -------------------------------------------------------------
# 4) View vs Copy — dikkat edilmesi gereken kenar durum
# -------------------------------------------------------------

def view_vs_copy() -> None:
    # Dilimleme NumPy'de varsayılan olarak *view* döner — yani aynı belleği paylaşır.
    # Bu, hızlıdır ama "ben sadece okudum sanmıştım" hatasına yol açabilir.
    asıl = np.arange(10)
    dilim = asıl[2:5]
    dilim[0] = 999

    # Asıl da değişti — çünkü dilim aynı belleğe işaret ediyor.
    assert asıl[2] == 999

    # Kopya istiyorsak explicit söylemeliyiz.
    güvenli = asıl[2:5].copy()
    güvenli[0] = -1
    assert asıl[2] == 999  # Bu sefer asıl etkilenmedi.


# -------------------------------------------------------------
# 5) Fancy ve boolean indexing
# -------------------------------------------------------------

def fancy_ve_boolean_indexing() -> None:
    # Fancy indexing: indeks dizisi vererek istediğimiz sırada eleman almak.
    # Top-k token seçerken veya örnek karıştırırken (shuffle) kullanılır.
    x = np.array([10, 20, 30, 40, 50])
    secim = x[[0, 2, 4]]
    assert secim.tolist() == [10, 30, 50]

    # Boolean indexing: bir koşul maskesi ile eleman süzmek.
    # Padding tokenlerini filtrelemek için tipik bir desen.
    maske = x > 20
    assert maske.tolist() == [False, False, True, True, True]
    assert x[maske].tolist() == [30, 40, 50]

    # Maskeyi atama hedefi olarak da kullanabiliriz.
    # Negatif değerleri sıfıra çekmek bunun klasik örneğidir (ReLU sezgisi).
    y = np.array([-1, 2, -3, 4])
    y[y < 0] = 0
    assert y.tolist() == [0, 2, 0, 4]


# -------------------------------------------------------------
# 6) Doğrudan çalıştırma
# -------------------------------------------------------------

if __name__ == "__main__":
    array_olusturma()
    kimlik_kartı()
    indeksleme()
    view_vs_copy()
    fancy_ve_boolean_indexing()
    print("02_numpy_array_temelleri: tüm assert'ler geçti.")
