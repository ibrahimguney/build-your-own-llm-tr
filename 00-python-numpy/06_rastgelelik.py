"""
06 — Rastgelelik ve Dağılımlar

Derin öğrenmenin neredeyse her aşaması bir yerde rastgele sayıya başvurur:
  - Ağırlıkları **rastgele başlatma** (init): tüm ağırlıklar sıfır olsa nöronlar
    simetrik kalır ve hiçbir şey öğrenemezler. Bu yüzden küçük rastgele değerlerle
    başlamak şarttır (Modül 06'da Xavier/Kaiming init'i göreceğiz).
  - Dropout: eğitim sırasında nöronları rastgele "sustur".
  - Sampling: dil modelinin ürettiği olasılık dağılımından rastgele token seç.
  - Shuffle: eğitim verisini her epoch'ta karıştır.

Önemli alışkanlık: **modern API olan `np.random.default_rng`** kullanılır.
Eski `np.random.rand` / `np.random.randn` global state'i değiştirir ve
parallel kodda hataya açıktır.
"""

import numpy as np


# -------------------------------------------------------------
# 1) Tekrarlanabilirlik: seed'lenmiş Generator
# -------------------------------------------------------------

def tekrarlanabilirlik() -> None:
    # Aynı seed → aynı çıktı. Bilimsel raporlamada deneyleri yeniden üretmek için kritik.
    rng_a = np.random.default_rng(seed=42)
    rng_b = np.random.default_rng(seed=42)

    x_a = rng_a.standard_normal(5)
    x_b = rng_b.standard_normal(5)
    assert np.array_equal(x_a, x_b)

    # Farklı seed → farklı çıktı.
    rng_c = np.random.default_rng(seed=7)
    x_c = rng_c.standard_normal(5)
    assert not np.array_equal(x_a, x_c)


# -------------------------------------------------------------
# 2) Sık kullanılan dağılımlar
# -------------------------------------------------------------

def dagilimlar() -> None:
    rng = np.random.default_rng(seed=0)

    # uniform: [0, 1) aralığında düzgün dağılım. "Rastgele bir olasılık üret" gibi.
    u = rng.random(size=10_000)
    # Beklenen ortalama 0.5; büyük örneklemde sağlanmalı.
    assert 0.49 < u.mean() < 0.51

    # standard_normal: N(0, 1) — sıfır ortalama, bir varyans.
    # Ağırlık initialization'da temel taşıdır.
    n = rng.standard_normal(size=10_000)
    assert -0.05 < n.mean() < 0.05
    assert 0.95 < n.std() < 1.05

    # normal: keyfi (mean, std). Xavier init'in basit hali için kullanılır.
    z = rng.normal(loc=0.0, scale=0.02, size=10_000)
    assert 0.018 < z.std() < 0.022


# -------------------------------------------------------------
# 3) Permütasyon — veriyi karıştırmak
# -------------------------------------------------------------

def permutasyon() -> None:
    # Her epoch'ta eğitim örneklerinin sırasını değiştirmek, modelin "veri sırasını
    # ezberlemesini" engeller ve daha hızlı genelleştirmesini sağlar.
    rng = np.random.default_rng(seed=1)
    indeksler = np.arange(10)
    karisik = rng.permutation(indeksler)

    # Aynı elemanlar, farklı sırada olmalı.
    assert sorted(karisik.tolist()) == list(range(10))
    assert not np.array_equal(karisik, indeksler)


# -------------------------------------------------------------
# 4) Init önemi — küçük bir gösterim
# -------------------------------------------------------------

def init_neden_onemli() -> None:
    """
    Eğer ağırlıkları çok büyük (örn. std=1) bir tek-katmanlı ağda kullanırsak,
    aktivasyon dağılımı patlar; çok küçük (örn. std=0.001) kullanırsak söner.
    İdeal, doğru ölçekli bir başlangıçtır.

    Bu bir önizleme; matematiksel detayı Modül 06'da.
    """
    rng = np.random.default_rng(seed=2)
    x = rng.standard_normal((1024, 64))   # girdi: 64 boyutlu, 1024 örnek

    W_buyuk = rng.standard_normal((64, 64)) * 1.0     # std=1: çok büyük
    W_kucuk = rng.standard_normal((64, 64)) * 0.001   # std=0.001: çok küçük
    # Önerilen: 1/sqrt(fan_in) ölçekli — burada fan_in=64.
    W_iyi = rng.standard_normal((64, 64)) * (1.0 / np.sqrt(64))

    h_buyuk = x @ W_buyuk
    h_kucuk = x @ W_kucuk
    h_iyi = x @ W_iyi

    # İyi init aktivasyon varyansını ~1 civarında tutar.
    # Diğer ikisi ya çok büyük ya çok küçük çıkar.
    assert h_buyuk.std() > 5.0
    assert h_kucuk.std() < 0.05
    assert 0.5 < h_iyi.std() < 1.5


# -------------------------------------------------------------
# 5) Kategorik örnekleme — dil modelinin token seçimi
# -------------------------------------------------------------

def kategorik_ornekleme() -> None:
    """
    Dil modelinin son katmanı her token için bir olasılık dağılımı üretir.
    Sonraki token'i bu dağılımdan örnekleyerek seçeriz.
    np.random.Generator.choice tam bunun için var.
    """
    rng = np.random.default_rng(seed=3)
    olasiliklar = np.array([0.7, 0.2, 0.1])  # 0 token'i çok olası
    secimler = rng.choice(3, size=10_000, p=olasiliklar)

    # Görülen frekanslar olasılıklara yakınsamalı.
    freq = np.bincount(secimler, minlength=3) / 10_000
    assert np.allclose(freq, olasiliklar, atol=0.02)


if __name__ == "__main__":
    tekrarlanabilirlik()
    dagilimlar()
    permutasyon()
    init_neden_onemli()
    kategorik_ornekleme()
    print("06_rastgelelik: tüm assert'ler geçti.")
