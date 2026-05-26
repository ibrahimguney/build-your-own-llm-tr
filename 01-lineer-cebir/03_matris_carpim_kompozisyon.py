"""
03 — Matris Çarpımı = Dönüşümlerin Kompozisyonu

Önceki dosyada matrisi "uzayın bir dönüşümü" olarak gördük. Şimdi soruyoruz:
İki matrisi çarptığımızda ne oluyor?

Cevap:
    (AB) @ x = A @ (B @ x)

Yani **önce B uygulanır, sonra A**. AB çarpımı, "B'den sonra A" dönüşümünü
tek bir matriste paketler. Bu, matris çarpımının neden "satır × sütun toplamı"
şeklinde tanımlandığını da açıklar — kompozisyonun çalışması için tek doğru
formül o.

İki sonuç:
  - AB ≠ BA genelde (rotasyon sonra shear ≠ shear sonra rotasyon).
  - Associativity: (AB)C = A(BC) ✓ (kompozisyon birleşmelidir).
  - Distributivity: A(B + C) = AB + AC ✓ (lineerliğin doğal sonucu).
"""

import numpy as np


# -------------------------------------------------------------
# 1) Kompozisyon — sırayla uygulamak ile çarpıp uygulamak eşittir
# -------------------------------------------------------------

def kompozisyon_esitligi() -> None:
    # Rotasyon (90°) ve ölçekleme (2x).
    R = np.array([[0.0, -1.0],
                  [1.0,  0.0]])         # 90° döndür
    S = np.array([[2.0, 0.0],
                  [0.0, 2.0]])          # 2x ölçek

    x = np.array([3.0, 4.0])

    # Yöntem 1: önce S uygula, sonra R uygula.
    sira = R @ (S @ x)

    # Yöntem 2: önce çarpımı hesapla, sonra uygula.
    M = R @ S   # "önce S, sonra R" matrisi
    tek_seferde = M @ x

    assert np.allclose(sira, tek_seferde)


# -------------------------------------------------------------
# 2) AB ≠ BA — sıranın önemi
# -------------------------------------------------------------

def degismeli_degil() -> None:
    R = np.array([[0.0, -1.0],
                  [1.0,  0.0]])  # 90° rotasyon
    Sh = np.array([[1.0, 1.0],
                   [0.0, 1.0]])  # x-shear

    AB = R @ Sh
    BA = Sh @ R

    # Sayısal olarak farklı olmalı.
    assert not np.allclose(AB, BA)

    # Geometrik sezgi:
    #   R @ Sh: önce shear sonra döndür → eğik bir paralelogramı 90° döndürürsün.
    #   Sh @ R: önce döndür sonra shear → döndürülmüş tabanı eğersin.
    # Sonuçlar gözle bakıldığında bile farklıdır.


# -------------------------------------------------------------
# 3) Identity matrisi — "hiçbir şey yapma" dönüşümü
# -------------------------------------------------------------

def identity_ozellikleri() -> None:
    I = np.eye(3)

    A = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0]])

    # IA = AI = A. Identity, çarpımda nötrdür (skalardaki 1 gibi).
    assert np.allclose(I @ A, A)
    assert np.allclose(A @ I, A)

    # Identity sütunları = standart taban vektörleri.
    # Bu, "matrisin sütunları tabanın görüntüsüdür" tezini destekler:
    # I tabana dokunmadığı için sütunları taban vektörleri olarak kalır.
    assert np.array_equal(I[:, 0], [1, 0, 0])
    assert np.array_equal(I[:, 1], [0, 1, 0])
    assert np.array_equal(I[:, 2], [0, 0, 1])


# -------------------------------------------------------------
# 4) Transpoz özdeşlikleri
# -------------------------------------------------------------

def transpoz_ozdeslikleri() -> None:
    rng = np.random.default_rng(0)
    A = rng.standard_normal((3, 4))
    B = rng.standard_normal((4, 2))

    # (AB)^T = B^T A^T — sıra ters döner. Attention'da Q K^T yazarken arka planda
    # bu kuralı kullanıyoruz: skor = Q K^T çünkü (K^T Q^T)^T = Q K^T şeklinde
    # transpoze edip okuyoruz.
    assert np.allclose((A @ B).T, B.T @ A.T)

    # (A^T)^T = A
    assert np.allclose(A.T.T, A)


# -------------------------------------------------------------
# 5) Birleşmelilik — (AB)C = A(BC)
# -------------------------------------------------------------

def birlesmelilik() -> None:
    """
    Kompozisyon birleşmelidir, çünkü "önce C, sonra B, sonra A" sırası nasıl
    parantezlersek parantezleyelim sırayı değiştirmez.

    Bu, performans için önemli: çok büyük matrislerde parantezi optimize
    ederek hesap maliyetini azaltırız (Modül 04 attention'da O(T²) vs O(T·D)
    seçimleri burada kök salar).
    """
    rng = np.random.default_rng(1)
    A = rng.standard_normal((4, 3))
    B = rng.standard_normal((3, 2))
    C = rng.standard_normal((2, 5))

    sol = (A @ B) @ C
    sag = A @ (B @ C)
    assert np.allclose(sol, sag)


# -------------------------------------------------------------
# 6) Çarpımdan satır/sütun okumak — kullanışlı sezgi
# -------------------------------------------------------------

def carpimdan_sutun_okumak() -> None:
    """
    (A B)'nin j. sütunu = A @ (B'nin j. sütunu).

    Bu sezgi, "matrisin sütunları tabanın görüntüsüdür" iddiasının
    matris çarpımı versiyonudur: B'nin sütunları "yeni taban"sa,
    AB'nin sütunları o yeni tabanın **A altında nereye gittiği**dir.
    """
    A = np.array([[1.0, 2.0],
                  [3.0, 4.0]])
    B = np.array([[5.0, 6.0],
                  [7.0, 8.0]])

    AB = A @ B

    # AB'nin 0. sütunu, A @ B[:, 0] olmalı.
    assert np.allclose(AB[:, 0], A @ B[:, 0])
    assert np.allclose(AB[:, 1], A @ B[:, 1])


if __name__ == "__main__":
    kompozisyon_esitligi()
    degismeli_degil()
    identity_ozellikleri()
    transpoz_ozdeslikleri()
    birlesmelilik()
    carpimdan_sutun_okumak()
    print("03_matris_carpim_kompozisyon: tüm assert'ler geçti.")
