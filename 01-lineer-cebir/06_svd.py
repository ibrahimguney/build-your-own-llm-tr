"""
06 — Singular Value Decomposition (SVD)

Modül 05'teki özdeğer ayrışımı (A = P D P⁻¹) yalnızca kare ve köşegenleştirilebilir
matrisler için çalışır. **SVD** ise her matrisi — kare olmasa, hatta tam ranklı
olmasa bile — evrensel olarak ayrıştırır:

    A = U Σ Vᵀ

Geometrik okuma çok güçlüdür:
    - Vᵀ: giriş uzayında bir **rotasyon** (yön bilgisi)
    - Σ:  eksen-bazlı **ölçekleme** (yalnızca uzatma/kısaltma)
    - U:  çıkış uzayında bir **rotasyon**

Yani her lineer dönüşüm aslında "döndür → ekseni ger → tekrar döndür" üçlüsüdür.
Bu yüzden SVD, ML'in evrensel ayrışım aracıdır: PCA, LSI, LoRA, embedding
sıkıştırma, attention skor matrisinin düşük rank analizi — hepsi aynı teoremden
beslenir.
"""

import numpy as np


# -------------------------------------------------------------
# 1) np.linalg.svd kullanımı — "ekonomik" mod
# -------------------------------------------------------------

def svd_temel_kullanim() -> None:
    """
    np.linalg.svd üç parça döndürür:
        U  : (m, m) ortogonal  — "tam" modda
        s  : (min(m,n),) tekil değerler (azalan sırada)
        Vt : (n, n) ortogonal — "tam" modda

    Pratikte `full_matrices=False` (ekonomik SVD) kullanılır: U (m, k) ve
    Vt (k, n) olur, burada k = min(m, n). Bu, gereksiz boyutları atar ve
    LLM ölçeğinde büyük tasarruf sağlar.
    """
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 3))  # dikdörtgen

    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    # k = min(m, n) = 3 olduğu için U (4,3), s (3,), Vt (3,3).
    assert U.shape == (4, 3)
    assert s.shape == (3,)
    assert Vt.shape == (3, 3)


# -------------------------------------------------------------
# 2) Yeniden inşa: U Σ Vᵀ = A
# -------------------------------------------------------------

def yeniden_insa() -> None:
    rng = np.random.default_rng(1)
    A = rng.standard_normal((5, 3))

    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    # s 1B bir dizidir; matris çarpımı için diag(s) gerekli.
    # diag, sıfır olmayan elemanları köşegene yerleştirir.
    Sigma = np.diag(s)
    yeniden = U @ Sigma @ Vt

    # Yeniden inşa edilen matris orijinaliyle sayısal toleransa kadar eşit olmalı.
    assert np.allclose(yeniden, A, atol=1e-10)


# -------------------------------------------------------------
# 3) U ve V'nin ortogonalliği — "yalnızca yön taşırlar"
# -------------------------------------------------------------

def ortogonallik() -> None:
    """
    U ve V ortogonal matrislerdir (full_matrices=False'ta yarı-ortogonal):
        Uᵀ U = I_k,   V Vᵀ = I_k

    Bu, "U ve V yalnızca rotasyon yapar, uzunluğu değiştirmez" sezgisinin
    cebirsel karşılığıdır. Tüm ölçek bilgisi Σ'da toplanır.
    """
    rng = np.random.default_rng(2)
    A = rng.standard_normal((6, 4))

    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    k = s.shape[0]  # = 4

    # U sütunları ortonormal: Uᵀ U = I.
    assert np.allclose(U.T @ U, np.eye(k), atol=1e-10)
    # Vt satırları ortonormal: Vt Vtᵀ = I (yani V'nin sütunları ortonormal).
    assert np.allclose(Vt @ Vt.T, np.eye(k), atol=1e-10)


# -------------------------------------------------------------
# 4) Tekil değerler ≥ 0 ve azalan sıralı
# -------------------------------------------------------------

def tekil_degerler_ozellikleri() -> None:
    """
    σ_i ≥ 0 (her zaman): tekil değer "uzunluk ölçeğidir", negatif olamaz.
    σ_1 ≥ σ_2 ≥ ... ≥ σ_k: konvansiyon olarak azalan sırada gelir.

    Bu sıralama, "en önemli yönler" sezgisinin temelidir: ilk birkaç σ matrisin
    enerjisinin büyük kısmını taşır; sonrakiler rafine detay. Bu özellik, bir
    sonraki dosyadaki düşük-rank yaklaşımın yapı taşıdır.
    """
    rng = np.random.default_rng(3)
    A = rng.standard_normal((5, 5))

    _, s, _ = np.linalg.svd(A, full_matrices=False)

    # Hepsi negatif olmayan.
    assert np.all(s >= 0)
    # Azalan sırada — np.diff negatif veya sıfır olmalı.
    assert np.all(np.diff(s) <= 0)


# -------------------------------------------------------------
# 5) Geometrik sezgi — birim çember elipse dönüşür
# -------------------------------------------------------------

def birim_cember_elipse() -> None:
    """
    Bir A matrisini birim çembere uyguladığımızda, sonuç bir elipstir.
    SVD bize bu elipsin yarı eksenlerini doğrudan verir:
        - Yarı eksen yönleri: U'nun sütunları
        - Yarı eksen uzunlukları: tekil değerler σ_i

    Yani SVD = matrisin gerçek geometrik şeklinin resmidir. Tek bir resmin
    içinde matrisin tüm önemli bilgisi var: yönelim + uzatma miktarları.
    """
    # 2x2 örnek — geometri net görülür.
    A = np.array([[3.0, 1.0],
                  [1.0, 2.0]])

    U, s, Vt = np.linalg.svd(A)

    # Birim çember üzerinde 200 nokta.
    theta = np.linspace(0, 2 * np.pi, 200)
    cember = np.stack([np.cos(theta), np.sin(theta)])  # shape (2, 200)

    # Dönüştür: her noktaya A uygula.
    elips = A @ cember

    # Sonuçtaki noktaların uzaklıkları arasında max = σ_1, min = σ_2 olmalı.
    # 200 nokta sürekli olmadığı için tolerans bırakıyoruz.
    uzakliklar = np.linalg.norm(elips, axis=0)
    assert np.isclose(np.max(uzakliklar), s[0], atol=1e-2)
    assert np.isclose(np.min(uzakliklar), s[1], atol=1e-2)


# -------------------------------------------------------------
# 6) Simetrik PSD → SVD ile özdeğer ayrışımı çakışır
# -------------------------------------------------------------

def simetrik_psd_svd_eig_baglantisi() -> None:
    """
    A simetrik **ve** pozitif yarı-kesin (PSD) ise:
        - özdeğerleri ≥ 0
        - özvektörleri ortogonal

    Bu durumda A = Q Λ Qᵀ formundaki özdeğer ayrışımı SVD'nin **tam olarak
    aynısıdır**: U = Q, Σ = Λ, V = Q. Modül 05'te eigh ile gördüğümüz şeyin,
    SVD'nin özel hali olduğunu burada sayısal olarak doğruluyoruz.
    """
    rng = np.random.default_rng(4)
    M = rng.standard_normal((4, 4))
    A = M @ M.T  # daima simetrik ve PSD (Modül 05'ten hatırla).

    # SVD (tekil değerler azalan sırada).
    U, s, Vt = np.linalg.svd(A)
    # Eigendecomposition (eigh artan sırada döner — ters çevirelim).
    lam, Q = np.linalg.eigh(A)
    lam_sira = lam[::-1]
    Q_sira = Q[:, ::-1]

    # Özdeğerler = tekil değerler.
    assert np.allclose(s, lam_sira, atol=1e-10)

    # U ve Q sütunlarının yönü aynı (işaret farkı serbest — birim vektörler
    # +v ile -v de aynı yönü gösterir). Mutlak değerle karşılaştırıyoruz.
    assert np.allclose(np.abs(U), np.abs(Q_sira), atol=1e-8)


# -------------------------------------------------------------
# 7) Rank = sıfır olmayan tekil değer sayısı
# -------------------------------------------------------------

def rank_svd_uzerinden() -> None:
    """
    Modül 04'te rank'i "bağımsız sütun sayısı" olarak tanıttık. SVD ile bunun
    **sayısal kesin** tanımı şudur:
        rank(A) = sıfırdan farklı σ_i sayısı.

    np.linalg.matrix_rank arka planda zaten SVD yapar ve küçük σ'ları sayısal
    gürültü olarak eler (varsayılan toleransla).
    """
    # Bilerek rank-2 olan bir 4x4 matris kuralım: (4×2) @ (2×4) = (4×4), rank ≤ 2.
    rng = np.random.default_rng(5)
    A = rng.standard_normal((4, 2))
    B = rng.standard_normal((2, 4))
    C = A @ B

    _, s, _ = np.linalg.svd(C, full_matrices=False)

    # İlk iki tekil değer anlamlı; son ikisi numerik gürültü düzeyinde.
    assert np.sum(s > 1e-10) == 2
    assert np.linalg.matrix_rank(C) == 2


if __name__ == "__main__":
    svd_temel_kullanim()
    yeniden_insa()
    ortogonallik()
    tekil_degerler_ozellikleri()
    birim_cember_elipse()
    simetrik_psd_svd_eig_baglantisi()
    rank_svd_uzerinden()
    print("06_svd: tüm assert'ler geçti.")
