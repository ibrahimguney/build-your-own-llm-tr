"""
03 — Broadcasting

Broadcasting, NumPy ve PyTorch'un kalbidir. "Farklı şekillerdeki tensörleri
sanki aynı şekildeymiş gibi işleme tabi tutmak" demektir.

Neden bu kadar önemli?
  - Batch boyutu (B) ile aynı işlemi tek tek değil topluca yaparız.
  - Bias eklerken bir (out_dim,) vektörü bir (B, out_dim) matrisine atarız.
  - Attention'da query-key skorlarını ölçeklerken `1 / sqrt(d_k)` skaler değeri
    bütün matrise yayarız.

Bu dosyada amaç: kuralı *ezberlemek* değil, ne zaman çalışıp çalışmayacağını
sezgiyle bilmek.
"""

import numpy as np


# -------------------------------------------------------------
# 1) Skalar ile dizi — en basit broadcast
# -------------------------------------------------------------

def skalar_yayilimi() -> None:
    # Bir skaleri, sanki array boyutuna sahipmiş gibi davranıyor.
    # Sayısal olarak: NumPy aslında belleği kopyalamıyor; "sanki kopyalanmış gibi" davranıyor.
    x = np.array([1.0, 2.0, 3.0])
    y = x * 2.0  # 2.0 "yayılıyor" → [2, 4, 6]
    assert y.tolist() == [2.0, 4.0, 6.0]

    # Standartlaştırma da bunun tipik bir uygulamasıdır:
    # her elemandan ortalamayı çıkar, standart sapmaya böl.
    veri = np.array([10.0, 12.0, 14.0, 16.0])
    standardize = (veri - veri.mean()) / veri.std()
    assert np.isclose(standardize.mean(), 0.0)
    assert np.isclose(standardize.std(), 1.0)


# -------------------------------------------------------------
# 2) Broadcasting kuralı — iki cümlede
# -------------------------------------------------------------
#
# Kural:
#   1) Şekilleri sağdan sola hizala.
#   2) Her eksende ya uzunluklar eşit olmalı, ya birinin uzunluğu 1 olmalı,
#      ya da o eksen taraflardan birinde hiç yoksa "1" gibi davranır.
#
# Eğer bir eksende uzunluk 1 ise, o eksen diğerinin uzunluğuna kadar tekrar
# ediliyormuş gibi davranır.
#
# Örnek hizalama:
#     A.shape = (    3, 4)
#     B.shape = (5,  1, 4)
#     ------------------
#     sonuç  = (5,  3, 4)
# -------------------------------------------------------------


def yayilim_kurali() -> None:
    # (3, 4) bir matrise (4,) şeklinde bir vektör eklemek — vektör her satıra ayrı ayrı eklenir.
    # Sinir ağında bias eklerken tam olarak bu desen.
    M = np.zeros((3, 4))
    bias = np.array([1.0, 2.0, 3.0, 4.0])
    sonuc = M + bias  # bias her satıra eklenir
    # Tüm satırlar aynı olmalı — çünkü M sıfır.
    assert sonuc.shape == (3, 4)
    assert np.allclose(sonuc[0], bias)
    assert np.allclose(sonuc[1], bias)


def yayilim_iki_yonlu() -> None:
    # Dış çarpım benzeri davranışı broadcasting ile inşa edebiliyoruz.
    # (3, 1) ile (1, 4) yayılınca → (3, 4) sonuç.
    # Bu, "her i için, her j için" demenin döngüsüz halidir.
    sutun = np.array([[1], [2], [3]])          # shape (3, 1)
    satir = np.array([[10, 20, 30, 40]])       # shape (1, 4)
    tablo = sutun + satir                      # shape (3, 4)

    beklenen = np.array([[11, 21, 31, 41],
                         [12, 22, 32, 42],
                         [13, 23, 33, 43]])
    assert tablo.shape == (3, 4)
    assert np.array_equal(tablo, beklenen)


# -------------------------------------------------------------
# 3) Broadcasting hatası — ne zaman çalışmaz?
# -------------------------------------------------------------

def yayilim_hatasi() -> None:
    # (3,) ile (4,) yayılamaz çünkü sağdan hizalanan eksenlerde uzunluklar farklı,
    # ve hiçbiri 1 değil.
    # NumPy bu durumda ValueError fırlatır. Bunu beklediğimizi göstermek için
    # try/except ile yakalıyoruz.
    a = np.array([1, 2, 3])
    b = np.array([1, 2, 3, 4])
    hata_yakalandi = False
    try:
        _ = a + b
    except ValueError:
        hata_yakalandi = True
    assert hata_yakalandi, "Broadcast hatası bekleniyordu ama hiç fırlamadı."


# -------------------------------------------------------------
# 4) Pratik desen: satır-bazlı normalizasyon
# -------------------------------------------------------------

def satir_bazli_normalizasyon() -> None:
    # 3 satır, her satırı kendi normuna bölelim → birim uzunlukta (norm=1) satırlar.
    # Bu, embedding vektörlerini "yön bilgisine" indirgemenin yoludur ve
    # kosinüs benzerliği hesabının ön adımıdır (sonraki dosyada göreceğiz).
    M = np.array([[3.0, 4.0],
                  [1.0, 0.0],
                  [0.0, 5.0]])
    normlar = np.linalg.norm(M, axis=1, keepdims=True)  # shape (3, 1)
    # keepdims=True olmasaydı şekil (3,) olur ve broadcasting tarafında
    # eksen hizalaması karışırdı. Bu detayı 05'te tekrar göreceğiz.
    birim_M = M / normlar
    # Her satırın uzunluğu artık 1 olmalı.
    assert np.allclose(np.linalg.norm(birim_M, axis=1), 1.0)


if __name__ == "__main__":
    skalar_yayilimi()
    yayilim_kurali()
    yayilim_iki_yonlu()
    yayilim_hatasi()
    satir_bazli_normalizasyon()
    print("03_broadcasting: tüm assert'ler geçti.")
