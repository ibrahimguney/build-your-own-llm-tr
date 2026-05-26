"""
05 — Özdeğer ve Özvektör

Sezgisel tanım:
  Bir A matrisi uzayı genelde **döndürür ve eğer** (shear). Ama bazı özel yönler
  vardır ki, A onları **sadece uzatır/kısaltır**, döndürmez. Bu özel yönlere
  **özvektör**, üzerinde uygulanan ölçek faktörüne ise **özdeğer** denir.

Denklemle:
  A v = λ v

Bu denklemi sağlayan sıfır olmayan v vektörü A'nın özvektörü, λ skaları
karşılık gelen özdeğerdir.

LLM bağlamı:
  - PCA (sıkıştırma) ve PageRank (Markov zincirinin durağan dağılımı), özdeğer
    problemlerinin doğrudan uygulamasıdır.
  - Transformer pretraining kararlılığını analizinde Hessian matrisinin
    özdeğer dağılımı analiz edilir.
  - Yarı-kesin (semidefinite) pozitif olmak özdeğerlerin işaretiyle tanımlanır.
"""

import numpy as np


# -------------------------------------------------------------
# 1) Tanım kontrolü: A v = λ v
# -------------------------------------------------------------

def tanim_kontrolu() -> None:
    # Diyagonal matris: doğal özvektörleri standart taban, özdeğerleri köşegen elemanları.
    A = np.array([[3.0, 0.0],
                  [0.0, 2.0]])

    # e1 = (1, 0) için A @ e1 = (3, 0) = 3 · e1 → özdeğer 3.
    e1 = np.array([1.0, 0.0])
    assert np.allclose(A @ e1, 3.0 * e1)

    # e2 = (0, 1) için A @ e2 = (0, 2) = 2 · e2 → özdeğer 2.
    e2 = np.array([0.0, 1.0])
    assert np.allclose(A @ e2, 2.0 * e2)


# -------------------------------------------------------------
# 2) Bir matrisin özdeğerlerini bulmak — np.linalg.eig
# -------------------------------------------------------------

def eig_kullanimi() -> None:
    """
    np.linalg.eig genel kare matrisler için (özdeğer, özvektör) çiftlerini döner.
    Özvektörler sütun olarak gelir; sıralanmaları garanti değildir.
    """
    A = np.array([[4.0, -2.0],
                  [1.0,  1.0]])

    ozdegerler, ozvektorler = np.linalg.eig(A)
    # Çıktıyı tek tek doğrulayalım: her sütun gerçekten özvektör mü?
    for i in range(2):
        v = ozvektorler[:, i]
        lam = ozdegerler[i]
        # A v ile λ v karşılaştırması: ikisi de aynı vektör olmalı.
        assert np.allclose(A @ v, lam * v, atol=1e-8)


# -------------------------------------------------------------
# 3) Karakteristik denklem — el ile özdeğer çıkarmak (2x2)
# -------------------------------------------------------------

def karakteristik_denklem() -> None:
    """
    A v = λ v denklemini (A - λI) v = 0 şekline getiririz. Bunun sıfır olmayan
    bir çözümü olması için (A - λI) tekil olmalı, yani:
        det(A - λI) = 0

    2x2'de bu, λ² - tr(A)·λ + det(A) = 0 olur. tr(A) = a11 + a22 köşegen
    toplamıdır ("trace"); aynı zamanda özdeğerlerin toplamına eşittir.

    Kontrol: özdeğerlerin toplamı = trace, çarpımı = determinant.
    """
    A = np.array([[4.0, -2.0],
                  [1.0,  1.0]])

    ozdegerler, _ = np.linalg.eig(A)

    assert np.isclose(np.sum(ozdegerler), np.trace(A))
    assert np.isclose(np.prod(ozdegerler), np.linalg.det(A))


# -------------------------------------------------------------
# 4) Simetrik matrisler — özel ve güzel davranış
# -------------------------------------------------------------

def simetrik_matris_ozellikleri() -> None:
    """
    Simetrik matrislerin (A = A^T) özellikleri özeldir:
      - Özdeğerleri **her zaman gerçeldir** (karmaşık değil).
      - Özvektörleri **birbirine ortogonaldir**.

    Bu yüzden simetrik matris özdeğer ayrışımı (eigendecomposition) SVD'nin
    özel bir durumudur. Sonraki dosyada bu bağlantıyı kuracağız.

    Pratik not: simetrik matrislerde np.linalg.eigh kullanılır — daha hızlı
    ve daha kararlıdır.
    """
    rng = np.random.default_rng(0)
    M = rng.standard_normal((4, 4))
    A = M @ M.T  # M·M^T daima simetrik ve pozitif yarı-kesindir

    # A simetrik mi? (sayısal toleransla)
    assert np.allclose(A, A.T)

    ozdegerler, ozvektorler = np.linalg.eigh(A)

    # Özdeğerler gerçel ve sıralı (eigh artan sıra döndürür).
    assert np.all(np.diff(ozdegerler) >= -1e-12)

    # Özvektörler birbirine dik mi? Q^T Q = I kontrolü.
    Q = ozvektorler
    assert np.allclose(Q.T @ Q, np.eye(4), atol=1e-8)


# -------------------------------------------------------------
# 5) Köşegenleştirme — A = P D P⁻¹
# -------------------------------------------------------------

def kosegenlestirme() -> None:
    """
    Bir matrisin n tane lineer bağımsız özvektörü varsa, onu köşegen bir
    matrise dönüştürebiliriz:

        A = P D P⁻¹

    Burada:
      - P sütunları özvektörler,
      - D köşegeninde özdeğerler.

    Bu, "A dönüşümüne özvektör koordinatlarında bakarsak, sadece bir köşegen
    ölçekleme görürüz" demektir. Yani **matrisin doğal koordinat sistemi vardır**.
    """
    A = np.array([[4.0, -2.0],
                  [1.0,  1.0]])

    lam, P = np.linalg.eig(A)
    D = np.diag(lam)

    yeniden = P @ D @ np.linalg.inv(P)
    assert np.allclose(yeniden, A, atol=1e-8)


# -------------------------------------------------------------
# 6) Markov zinciri — özdeğerin pratik gücü
# -------------------------------------------------------------

def markov_duragan_dagilim() -> None:
    """
    Bir geçiş matrisi P (her satır olasılık dağılımı) verildiğinde, **durağan
    dağılım** π = πP koşulunu sağlar — yani P'nin özdeğeri 1 olan özvektörüdür.

    Bu, PageRank algoritmasının ve daha genel olarak Markov zincirleri tabanlı
    örnekleme metodlarının (MCMC) kalbidir. LLM dünyasında doğrudan kullanılmasa
    bile sezgisi yeni başlayan için değerli.

    Örnek: 3 durumlu hava — Güneşli (G), Bulutlu (B), Yağmurlu (Y).
    Geçiş matrisi P satırlarda "şu anki durumdan diğerlerine olasılık".
    """
    P = np.array([[0.8, 0.15, 0.05],
                  [0.2, 0.6,  0.2 ],
                  [0.1, 0.3,  0.6 ]])

    # Tüm satırlar olasılık dağılımı olmalı.
    assert np.allclose(P.sum(axis=1), 1.0)

    # π P = π → π, P^T'nin 1 özdeğerine ait özvektörü.
    # (Çünkü πP = π'yi sütun haline çevirirsek P^T π^T = π^T olur.)
    lam, V = np.linalg.eig(P.T)

    # Özdeğer = 1'e en yakın olanı seç.
    idx = np.argmin(np.abs(lam - 1.0))
    pi = np.real(V[:, idx])
    pi = pi / pi.sum()  # olasılık dağılımı haline getir (toplamı 1)

    # Tekrarlı çarpımdan da varmamız gereken yer aynı olmalı.
    # Başlangıçta tek bir durumdan başlayıp 1000 adım gidersek π'ye yaklaşırız.
    durum = np.array([1.0, 0.0, 0.0])
    for _ in range(1000):
        durum = durum @ P

    assert np.allclose(durum, pi, atol=1e-6)


if __name__ == "__main__":
    tanim_kontrolu()
    eig_kullanimi()
    karakteristik_denklem()
    simetrik_matris_ozellikleri()
    kosegenlestirme()
    markov_duragan_dagilim()
    print("05_ozdeger_ozvektor: tüm assert'ler geçti.")
