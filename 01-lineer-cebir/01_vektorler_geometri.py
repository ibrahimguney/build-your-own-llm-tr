"""
01 — Vektör Geometrisi

Modül 00'da vektörü "NumPy array" olarak ele aldık. Burada onu **uzayda bir ok**
olarak görmeye başlıyoruz. Bu zihinsel model olmadan matrisleri "lineer dönüşüm"
olarak okumak mümkün değil.

Sorular:
  - Vektör toplama / skalar çarpım geometrik olarak ne yapar?
  - İki vektörün **iç çarpımı** neden açıyla ilgili?
  - Bir vektörü başka bir vektörün üzerine **projekte etmek** ne demek?
  - Bunların attention skorlarıyla ne ilgisi var?
"""

import numpy as np


# -------------------------------------------------------------
# 1) Vektör toplama ve skalar çarpım — paralelogram kuralı
# -------------------------------------------------------------

def toplam_ve_skalar() -> None:
    # İki vektörün toplamı, "paralelogram"ın diyagonalini verir.
    # Ya da: u'nun ucundan v'ye gidersek, varış noktası u+v olur.
    u = np.array([3.0, 1.0])
    v = np.array([1.0, 2.0])

    toplam = u + v
    assert np.array_equal(toplam, np.array([4.0, 3.0]))

    # Skalar çarpım vektörün **yönünü değiştirmez**, yalnızca **uzunluğunu** ölçekler.
    # Negatif skalar yönü tersine çevirir.
    iki_u = 2.0 * u
    ters_u = -1.0 * u
    assert np.array_equal(iki_u, np.array([6.0, 2.0]))
    assert np.array_equal(ters_u, np.array([-3.0, -1.0]))


# -------------------------------------------------------------
# 2) Norm — bir vektörün uzunluğu
# -------------------------------------------------------------

def norm_ve_birimlestirme() -> None:
    # L2 norm: sqrt(Σ x_i^2). Öklid uzunluğu.
    v = np.array([3.0, 4.0])
    assert np.isclose(np.linalg.norm(v), 5.0)  # 3-4-5 üçgeni.

    # Birim vektör: kendi normuna bölünmüş vektör. Uzunluğu 1, sadece yön taşır.
    # Embedding'lerde yön bilgisini izole etmenin ana yöntemi budur.
    birim = v / np.linalg.norm(v)
    assert np.isclose(np.linalg.norm(birim), 1.0)


# -------------------------------------------------------------
# 3) İç çarpım — geometrik tanım
# -------------------------------------------------------------
#
# u · v = |u| · |v| · cos(theta)
#
# Yani iç çarpım sadece "elemanları çarpıp topla" değil; o sonucun
# geometrik anlamı "iki vektörün ne kadar aynı yöne baktığı".
#
# Sonuç pozitifse → aynı yarı düzlemdeler (dar açı).
# Sonuç sıfırsa  → dik.
# Sonuç negatifse → zıt yarı düzlemde (geniş açı).
# -------------------------------------------------------------


def ic_carpim_geometrisi() -> None:
    u = np.array([1.0, 0.0])
    # 45 derece eğimli bir vektör — köşesi (1, 1)'de.
    v = np.array([1.0, 1.0])

    ic = np.dot(u, v)
    cos_theta = ic / (np.linalg.norm(u) * np.linalg.norm(v))

    # cos(45°) = sqrt(2)/2 ≈ 0.7071
    assert np.isclose(cos_theta, np.sqrt(2) / 2)


def dik_vektorler() -> None:
    # Dik (ortogonal) vektörlerin iç çarpımı tam olarak 0'dır.
    # "Bağımsız bilgi" sezgisinin matematiksel karşılığı budur.
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    assert np.isclose(np.dot(e1, e2), 0.0)


# -------------------------------------------------------------
# 4) Projeksiyon — bir vektörün başkası üzerindeki "gölgesi"
# -------------------------------------------------------------
#
# u vektörünün v üzerine projeksiyonu:
#
#   proj_v(u) = ((u · v) / (v · v)) * v
#
# Bu, u'nun **v doğrultusundaki bileşeni**dir. Geri kalan u - proj_v(u)
# vektörü v'ye diktir.
#
# Attention'da bir token'in başka bir token'a ne kadar "dikkat verdiği"
# tam olarak bu projeksiyon mantığına dayanır: Q[i] vektörünün K[j] yönündeki
# bileşeni ne kadar büyükse, j'ye o kadar dikkat verilir.
# -------------------------------------------------------------


def projeksiyon(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """u'nun v üzerine ortogonal projeksiyonu."""
    # v'nin norm karesini paydada kullanıyoruz; v birim vektör ise basitleşir.
    return (np.dot(u, v) / np.dot(v, v)) * v


def projeksiyon_testi() -> None:
    u = np.array([3.0, 4.0])
    # x-ekseni üzerine projeksiyon → sadece x bileşeni kalır.
    p = projeksiyon(u, np.array([1.0, 0.0]))
    assert np.allclose(p, [3.0, 0.0])

    # u'dan projeksiyonu çıkardığımızda kalan vektör projeksiyon eksenine dik olmalı.
    v = np.array([1.0, 1.0])
    p = projeksiyon(u, v)
    kalan = u - p
    # kalan · v = 0 → diktir
    assert np.isclose(np.dot(kalan, v), 0.0)


# -------------------------------------------------------------
# 5) Lineer kombinasyon — vektörlerin "karışımı"
# -------------------------------------------------------------

def lineer_kombinasyon() -> None:
    # 2 boyutlu uzayda, lineer bağımsız iki vektör (taban) tüm uzayı kapsar.
    # Yani başka herhangi bir vektörü α·e1 + β·e2 olarak yazabiliriz.
    # Bu sezgi, "matrisin sütunları taban vektörlerin görüntüsüdür" iddiamızın temelidir.
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])

    hedef = np.array([3.0, 5.0])
    yeniden = 3.0 * e1 + 5.0 * e2
    assert np.array_equal(hedef, yeniden)

    # İki vektör paralelse (örn. [1,0] ve [2,0]), bütün 2B uzayı kapsayamazlar;
    # sadece x ekseninde kalırlar. Lineer **bağımlı** olurlar — taban değildirler.


if __name__ == "__main__":
    toplam_ve_skalar()
    norm_ve_birimlestirme()
    ic_carpim_geometrisi()
    dik_vektorler()
    projeksiyon_testi()
    lineer_kombinasyon()
    print("01_vektorler_geometri: tüm assert'ler geçti.")
