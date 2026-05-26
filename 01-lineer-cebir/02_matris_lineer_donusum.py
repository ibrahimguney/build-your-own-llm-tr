"""
02 — Matris = Lineer Dönüşüm

Bu, modülün **en önemli** dosyası. Buradaki sezgiyi içselleştirirsen geri kalan
her şey (özdeğer, SVD, LoRA) doğal olarak yerine oturur.

Ana fikir:
  Bir matris, uzayın bir **lineer dönüşümüdür**. Yani:
    - orijini sabit tutar,
    - paralel çizgileri paralel tutar,
    - eşit aralıklı noktaları eşit aralıklı tutar.

Daha güçlü fikir:
  Bir matrisin **sütunları**, standart taban vektörlerin (e_1, e_2, ...) dönüşüm
  sonrası gittiği yerlerdir.

  A @ e_1 = A'nın 1. sütunu
  A @ e_2 = A'nın 2. sütunu

  Yani matrisi yazarken aslında "tabanın nereye gideceğini" yazıyoruz.
"""

import numpy as np


# -------------------------------------------------------------
# 1) Sütun-eşittir-tabanın-görüntüsü ilişkisi
# -------------------------------------------------------------

def sutun_taban_iliskisi() -> None:
    # Bir matris yazalım: A. Onun sütunları neye benziyor görelim.
    A = np.array([[2.0, 1.0],
                  [0.0, 3.0]])

    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])

    # A @ e1, A'nın 1. sütununu vermeli.
    assert np.array_equal(A @ e1, A[:, 0])
    # A @ e2, A'nın 2. sütununu vermeli.
    assert np.array_equal(A @ e2, A[:, 1])

    # Sezgi: A matrisini görüp "1. sütun [2, 0], demek ki e1 → [2, 0]'a gidiyor" diyebiliriz.
    # Yani 2 birim sağa giden e1, dönüşüm sonrası 2 birim sağa gitmeye devam ediyor (ölçek 2).
    # 1 birim yukarı giden e2 ise dönüşüm sonrası [1, 3]'e gidiyor — hem sağa kaydı hem yukarı uzadı.


# -------------------------------------------------------------
# 2) Klasik 2B dönüşümler: ölçekleme
# -------------------------------------------------------------

def olcekleme_matrisi() -> None:
    # Diyagonal matris her ekseni bağımsız ölçekler.
    # diag(sx, sy): x ekseninde sx, y ekseninde sy katı.
    S = np.array([[2.0, 0.0],
                  [0.0, 0.5]])

    v = np.array([1.0, 1.0])
    sonuc = S @ v
    # x ekseni 2x, y ekseni 0.5x.
    assert np.array_equal(sonuc, np.array([2.0, 0.5]))


# -------------------------------------------------------------
# 3) Rotasyon
# -------------------------------------------------------------

def rotasyon_matrisi(theta_rad: float) -> np.ndarray:
    """
    Saat yönünün tersine theta_rad açısı kadar döndüren 2B matris.

    R = [[cos θ, -sin θ],
         [sin θ,  cos θ]]

    Sütunlara bak:
      1. sütun = (cos θ, sin θ) → bu, e1'in döndürülmüş hali (birim çember üzerinde θ).
      2. sütun = (-sin θ, cos θ) → bu da e2'nin döndürülmüş hali (θ + 90°).
    Yani "rotasyon" denen şey aslında "tabanı birim çember üzerinde döndürmek".
    """
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s],
                     [s, c]])


def rotasyon_testi() -> None:
    R90 = rotasyon_matrisi(np.pi / 2)

    e1 = np.array([1.0, 0.0])
    # 90° döndürülmüş e1 → (0, 1) olmalı.
    assert np.allclose(R90 @ e1, np.array([0.0, 1.0]))

    # Rotasyon uzunlukları korur: norm aynı kalmalı.
    v = np.array([3.0, 4.0])
    assert np.isclose(np.linalg.norm(R90 @ v), np.linalg.norm(v))


# -------------------------------------------------------------
# 4) Shear (kaydırma) — paralelogram yapan dönüşüm
# -------------------------------------------------------------

def shear_matrisi() -> None:
    # x-shear: y koordinatı x'e karışır, y kendisi değişmez.
    # Birim kare → paralelogram olur. Üst kenar yana kayar, alt kenar yerinde durur.
    Sh = np.array([[1.0, 1.0],
                   [0.0, 1.0]])

    # Alt-sol köşe (0, 0) → (0, 0) — değişmez.
    # Alt-sağ köşe (1, 0) → (1, 0) — değişmez (çünkü y=0).
    # Üst-sol köşe (0, 1) → (1, 1) — sağa kaydı.
    # Üst-sağ köşe (1, 1) → (2, 1) — daha çok sağa kaydı.
    assert np.array_equal(Sh @ np.array([1.0, 0.0]), [1.0, 0.0])
    assert np.array_equal(Sh @ np.array([0.0, 1.0]), [1.0, 1.0])


# -------------------------------------------------------------
# 5) Aynalama (reflection)
# -------------------------------------------------------------

def aynalama() -> None:
    # x ekseni üzerinde aynalama: y koordinatının işareti değişir.
    M = np.array([[1.0, 0.0],
                  [0.0, -1.0]])

    assert np.array_equal(M @ np.array([3.0, 4.0]), [3.0, -4.0])

    # Aynalama özel bir özelliği taşır: kendisiyle iki kez uygulanması identity verir.
    # Yani M @ M = I.
    assert np.allclose(M @ M, np.eye(2))


# -------------------------------------------------------------
# 6) Birim kareyi dönüştürmek — görsel sezgi (sayısal)
# -------------------------------------------------------------

def birim_kareyi_donustur() -> None:
    """
    Birim karenin 4 köşesini bir matrise uygulayıp nereye gittiklerini hesaplayalım.
    Notebook'ta bunu görselle göreceğiz; burada sayısal olarak doğrularız.
    """
    A = np.array([[1.0, 1.0],
                  [0.5, 1.5]])

    koseler = np.array([[0, 1, 1, 0],   # x koordinatları
                        [0, 0, 1, 1]])  # y koordinatları → shape (2, 4)

    # Her köşe vektörünü A ile çarpalım. Tüm köşeler için tek matmul:
    donusmus = A @ koseler  # shape (2, 4)

    # Köşeler hala paralelogram köşeleri oluşturmalı:
    # Karşı kenarlar paralel (vektör olarak fark eşit) olmalıdır.
    kenar1 = donusmus[:, 1] - donusmus[:, 0]   # alt kenar
    kenar3 = donusmus[:, 2] - donusmus[:, 3]   # üst kenar
    assert np.allclose(kenar1, kenar3)


if __name__ == "__main__":
    sutun_taban_iliskisi()
    olcekleme_matrisi()
    rotasyon_testi()
    shear_matrisi()
    aynalama()
    birim_kareyi_donustur()
    print("02_matris_lineer_donusum: tüm assert'ler geçti.")
