"""
04 — Determinant ve Ters

Bir dönüşüm uzayı sıkıştırıyor mu, genişletiyor mu, yoksa tersine mi çeviriyor?
Bu sorulara **determinant** cevap verir.

Determinantın geometrik anlamı:
  - |det A| = A dönüşümünün **alanı/hacmi ne kadar ölçeklediği**.
  - det A > 0: yönelim korunur (saat yönü saat yönü kalır).
  - det A < 0: yönelim ters döner (aynalama içerir).
  - det A = 0: dönüşüm bir boyuta sıkışır → **geri alınamaz**, yani ters yoktur.

Ve ters:
  - A · A⁻¹ = I
  - A⁻¹ var ⟺ det A ≠ 0
  - Ters varsa A "tersinir" (invertible) ya da "tekil olmayan" (non-singular) denir.
"""

import numpy as np


# -------------------------------------------------------------
# 1) Determinant = alan ölçeği (2B'de)
# -------------------------------------------------------------

def determinant_alan() -> None:
    # 2x ölçekleme matrisi her yönü 2x büyütür. Alan 4x büyümeli.
    S2 = np.array([[2.0, 0.0],
                   [0.0, 2.0]])
    assert np.isclose(np.linalg.det(S2), 4.0)

    # Rotasyon alanı değiştirmez → determinant 1.
    theta = 1.2
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    assert np.isclose(np.linalg.det(R), 1.0)

    # x ekseni üzerinde aynalama → yönelim tersine döner → det = -1.
    M = np.array([[1.0, 0.0],
                  [0.0, -1.0]])
    assert np.isclose(np.linalg.det(M), -1.0)


# -------------------------------------------------------------
# 2) det = 0 → tekil matris → ters yok
# -------------------------------------------------------------

def tekil_matris() -> None:
    # İki paralel sütun (lineer bağımlı sütunlar) → dönüşüm bir doğruya çöker.
    # Hiçbir 2B noktadan bu doğru dışına çıkamayız → bilgi kaybı → geri dönüş yok.
    A = np.array([[1.0, 2.0],
                  [2.0, 4.0]])  # 2. satır = 2 × 1. satır

    assert np.isclose(np.linalg.det(A), 0.0)

    # Ters almayı denesek hata fırlar. Burada bunu beklediğimizi gösteriyoruz.
    hata = False
    try:
        np.linalg.inv(A)
    except np.linalg.LinAlgError:
        hata = True
    assert hata


# -------------------------------------------------------------
# 3) Tersin tanımı: A · A⁻¹ = I
# -------------------------------------------------------------

def ters_tanim() -> None:
    A = np.array([[4.0, 3.0],
                  [6.0, 3.0]])
    # det A = 12 - 18 = -6, sıfırdan farklı → ters var.
    assert not np.isclose(np.linalg.det(A), 0.0)

    A_inv = np.linalg.inv(A)
    # Ters çarpımının her iki yönde de identity vermesi gerekir.
    assert np.allclose(A @ A_inv, np.eye(2))
    assert np.allclose(A_inv @ A, np.eye(2))


# -------------------------------------------------------------
# 4) Determinant özdeşlikleri — neden işe yarar?
# -------------------------------------------------------------

def determinant_ozdeslikleri() -> None:
    rng = np.random.default_rng(0)
    A = rng.standard_normal((3, 3))
    B = rng.standard_normal((3, 3))

    # det(AB) = det(A) · det(B). Çünkü iki ölçekleme oranı çarpılır.
    assert np.isclose(np.linalg.det(A @ B),
                      np.linalg.det(A) * np.linalg.det(B))

    # det(A^T) = det(A). Transpoz sütun-satır takası yapar, alan ölçeği aynı kalır.
    assert np.isclose(np.linalg.det(A.T), np.linalg.det(A))

    # det(A^-1) = 1 / det(A). Tersinir ölçek 2x ise tersine alanı 0.5x'e indirir.
    if abs(np.linalg.det(A)) > 1e-6:
        assert np.isclose(np.linalg.det(np.linalg.inv(A)),
                          1.0 / np.linalg.det(A))


# -------------------------------------------------------------
# 5) Lineer denklem çözmek — Ax = b
# -------------------------------------------------------------

def lineer_sistem() -> None:
    """
    A tersinir ise Ax = b denkleminin tek bir çözümü vardır: x = A⁻¹ b.

    Pratikte np.linalg.inv kullanmak yerine **np.linalg.solve** tercih edilir;
    çünkü doğrudan çözüm hem hızlı hem daha kararlıdır (matrisi tersine
    çevirmek sayısal hatayı şişirir). Ama tanım netliği için ikisini de göstereceğiz.
    """
    A = np.array([[3.0, 1.0],
                  [1.0, 2.0]])
    b = np.array([9.0, 8.0])

    # Yöntem 1: doğrudan çöz (tercih edilen)
    x = np.linalg.solve(A, b)

    # Yöntem 2: ters al, çarp (öğretici ama pratikte tercih edilmeyen)
    x_alt = np.linalg.inv(A) @ b

    assert np.allclose(x, x_alt)
    # Çözüm sağlama: Ax = b mi?
    assert np.allclose(A @ x, b)


# -------------------------------------------------------------
# 6) Rank — "kaç bağımsız sütun var?"
# -------------------------------------------------------------

def rank_sezgisi() -> None:
    """
    Rank, bir matrisin "kaç boyutluk bilgi taşıdığı"dır.
      - Tam ranklı (n×n için rank=n) → tersinir, det ≠ 0.
      - Rank eksik → bilgi kaybı, det = 0, ters yok.

    Bu sezgi 06'da SVD ile sıfır olmayan tekil değer sayısına eşit hale gelecek
    ve 07'de düşük-rank yaklaşımın kalbini oluşturacak.
    """
    tam_ranklı = np.array([[1.0, 0.0],
                           [0.0, 1.0]])
    assert np.linalg.matrix_rank(tam_ranklı) == 2

    rank_eksik = np.array([[1.0, 2.0],
                           [2.0, 4.0]])  # 2. satır = 2 × 1. satır
    assert np.linalg.matrix_rank(rank_eksik) == 1


if __name__ == "__main__":
    determinant_alan()
    tekil_matris()
    ters_tanim()
    determinant_ozdeslikleri()
    lineer_sistem()
    rank_sezgisi()
    print("04_determinant_ters: tüm assert'ler geçti.")
