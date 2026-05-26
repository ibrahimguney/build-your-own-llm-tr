"""
05 — Eksenler, reshape, transpose

LLM kodlarında "axis=?" sorusunun cevabını bilmek, için-dışında olmanın eşiğidir.
Bu dosyada `axis` argümanını ve şekil değiştirme operasyonlarını tek bir yerde toparlıyoruz.

Anahtar sezgi:
  - "axis=k" demek "k. ekseni *yok edecek/küçültecek* şekilde işle" demektir.
  - axis=0: ilk eksen yok olur → her sütun için ayrı bir sonuç çıkar.
  - axis=1: ikinci eksen yok olur → her satır için ayrı bir sonuç çıkar.
  - axis=-1: son eksen yok olur — bu, Transformer'da en sık görüleceğimiz formdur
    (örn. softmax(logits, dim=-1)).
"""

import numpy as np


# -------------------------------------------------------------
# 1) axis argümanı — "hangi eksen yok oluyor?"
# -------------------------------------------------------------

def axis_sezgisi() -> None:
    M = np.array([[1, 2, 3],
                  [4, 5, 6]])  # shape (2, 3)

    # Tüm elemanların toplamı: axis=None (varsayılan).
    assert M.sum() == 21

    # axis=0 → ilk eksen (satırlar) yok olur, her sütun için ayrı toplam.
    # Sonucun şekli (3,) — yani sütun sayısı kadar.
    sutun_toplamlari = M.sum(axis=0)
    assert sutun_toplamlari.shape == (3,)
    assert sutun_toplamlari.tolist() == [5, 7, 9]

    # axis=1 → ikinci eksen (sütunlar) yok olur, her satır için ayrı toplam.
    # Sonucun şekli (2,).
    satir_toplamlari = M.sum(axis=1)
    assert satir_toplamlari.shape == (2,)
    assert satir_toplamlari.tolist() == [6, 15]


# -------------------------------------------------------------
# 2) keepdims — boyutu korumak
# -------------------------------------------------------------

def keepdims_neden_onemli() -> None:
    # Yukarıda axis=1 toplamı (2,) şekilli bir vektör verdi.
    # Bunu orijinal M'e bölmek istesek broadcasting tarafında sıkıntı çıkar:
    #   M.shape = (2, 3),  toplam.shape = (2,) → hizalanmaz!
    # Çözüm: keepdims=True ile toplamın şeklini (2, 1) tutmak.
    M = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

    satir_toplamlari = M.sum(axis=1, keepdims=True)
    assert satir_toplamlari.shape == (2, 1)

    # Şimdi her satırı kendi toplamına bölebiliriz → satır olasılığa dönüşür.
    # Bu, sıfırdan softmax yazarken yapacağımız şeyin ta kendisi.
    olasiliklar = M / satir_toplamlari
    assert np.allclose(olasiliklar.sum(axis=1), 1.0)


# -------------------------------------------------------------
# 3) reshape — toplam eleman sayısı sabit kaldıkça serbest
# -------------------------------------------------------------

def reshape_ornegi() -> None:
    # arange(12) → 12 eleman. Bunu (3, 4) ya da (2, 6) ya da (2, 2, 3) yapabiliriz.
    x = np.arange(12)
    M = x.reshape(3, 4)
    assert M.shape == (3, 4)

    # -1 "kalanı sen hesapla" demektir; ihtiyaç duyduğumuzda işimizi kolaylaştırır.
    # Transformer'da (B, T, C) tensörünü (B*T, C) yapmanın deyimsel yolu budur.
    y = M.reshape(-1, 2)
    assert y.shape == (6, 2)

    # Sırayla eleman okuma: reshape "C order" (satır bazlı, yani son eksen en hızlı değişir)
    # varsayılanını kullanır. Yani x[0..3] M[0]'a, x[4..7] M[1]'e gider.
    assert M[1, 0] == 4


# -------------------------------------------------------------
# 4) transpose / .T — eksenleri yer değiştirmek
# -------------------------------------------------------------

def transpose_ornegi() -> None:
    # 2D'de .T en sade kullanım.
    M = np.arange(6).reshape(2, 3)  # (2, 3)
    assert M.T.shape == (3, 2)

    # 3D ve üzerinde, eksenleri tek tek belirtmek için np.transpose'u kullanırız.
    # Multi-head attention'da (B, T, H, D_h) ↔ (B, H, T, D_h) dönüşümünü tam
    # böyle yaparız: head ekseni başa çekilir ki "her head ayrı bir batch gibi" işlensin.
    T_BxHxTxD = np.arange(2 * 2 * 3 * 4).reshape(2, 3, 2, 4)  # (B=2, T=3, H=2, D_h=4)
    permuted = T_BxHxTxD.transpose(0, 2, 1, 3)                # (B, H, T, D_h)
    assert permuted.shape == (2, 2, 3, 4)


# -------------------------------------------------------------
# 5) Pratik desen: sütun-bazlı normalizasyon (döngüsüz)
# -------------------------------------------------------------

def sutun_normalizasyonu() -> None:
    """
    Bir (N, D) veri matrisinin her D-boyutlu sütununu kendi ortalamasıyla
    merkezleyip kendi standart sapmasına böleriz.

    Bu, "feature scaling" denilen klasik adımdır ve sıfırdan yazıldığında
    eksen düşüncesinin doğrudan testidir.
    """
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5)) * np.array([1.0, 10.0, 0.1, 5.0, 2.0]) + 3.0

    # axis=0 → ilk eksen yok olur, her sütun için bir mean/std değeri çıkar.
    # keepdims=True ile (1, 5) şekli koruyup orijinal X'e (100, 5) yayabiliriz.
    ortalama = X.mean(axis=0, keepdims=True)  # (1, 5)
    std = X.std(axis=0, keepdims=True)        # (1, 5)

    Z = (X - ortalama) / std

    # Doğrulama: her sütunun ortalaması ~0, std'si ~1 olmalı.
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(Z.std(axis=0), 1.0, atol=1e-8)


if __name__ == "__main__":
    axis_sezgisi()
    keepdims_neden_onemli()
    reshape_ornegi()
    transpose_ornegi()
    sutun_normalizasyonu()
    print("05_eksenler_ve_reshape: tüm assert'ler geçti.")
