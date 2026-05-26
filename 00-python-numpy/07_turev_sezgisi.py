"""
07 — Türev Sezgisi

Otomatik türev (autograd) modül 02'de PyTorch'la geliyor. Ama o "sihir" gibi
gelmeden önce, türevin kendisini sayısal olarak hissetmek istiyoruz.

Soracağımız sorular:
  - Türev nedir? Bir noktada fonksiyonun "eğimi".
  - Sayısal türev nasıl hesaplanır? `(f(x+h) - f(x-h)) / (2h)`.
  - Çok değişkenli fonksiyonun gradyanı nedir?
  - Neden gradyanın *zıt* yönünde adım atarız (gradient descent)?

Bu dosyayı bitirdiğinde, `loss.backward()` çağrısının arkasında ne döndüğüne
dair zihinsel bir model edinmiş olacaksın.
"""

import numpy as np


# -------------------------------------------------------------
# 1) Tek değişkenli sayısal türev
# -------------------------------------------------------------

def sayisal_turev(f, x: float, h: float = 1e-5) -> float:
    """
    Merkezi fark (central difference) yöntemi:
        f'(x) ≈ (f(x+h) - f(x-h)) / (2h)

    Neden ileri farkın (`(f(x+h) - f(x)) / h`) yerine merkezi farkı kullanıyoruz?
    Merkezi fark, ikinci dereceden doğrudur — yani h küçüldükçe hata h^2 ile
    azalır. İleri fark sadece birinci dereceden doğrudur. Sayısal kararlılık
    açısından net bir kazanım.

    h çok küçük olursa (örn. 1e-15) floating point yuvarlamadan dolayı hata
    *artar*. 1e-5 dengeli bir seçimdir.
    """
    return (f(x + h) - f(x - h)) / (2 * h)


def tek_degiskenli_turev() -> None:
    # f(x) = x^2 → f'(x) = 2x
    f = lambda x: x ** 2
    # x=3 noktasında f'(3) = 6 olmalı.
    assert np.isclose(sayisal_turev(f, 3.0), 6.0, atol=1e-6)

    # f(x) = sin(x) → f'(x) = cos(x)
    g = lambda x: np.sin(x)
    # x=0'da türev cos(0) = 1.
    assert np.isclose(sayisal_turev(g, 0.0), 1.0, atol=1e-6)


# -------------------------------------------------------------
# 2) Çok değişkenli gradyan
# -------------------------------------------------------------

def sayisal_gradyan(f, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Bir f: R^n → R fonksiyonunun gradyanı:
        ∇f(x)[i] = ∂f/∂x_i

    Her boyuttaki kısmi türevi ayrı ayrı sayısal olarak hesaplıyoruz.
    Bu yöntem yavaştır (n boyutluda 2n fonksiyon çağrısı) — autograd asıl gücünü
    burada gösterir: tek bir geri-yayılım hepsini hesaplayabilir.
    Yine de gradyanın *ne* olduğunu kavramak için doğru zihinsel modeldir.
    """
    grad = np.zeros_like(x, dtype=float)
    # np.nditer ile çok boyutlu bir tensörün her elemanını gezeceğiz.
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        ix = it.multi_index
        eski = x[ix]

        x[ix] = eski + h
        arti = f(x)
        x[ix] = eski - h
        eksi = f(x)
        x[ix] = eski  # geri yükle; aksi halde sonraki adımı kirletir

        grad[ix] = (arti - eksi) / (2 * h)
        it.iternext()
    return grad


def cok_degiskenli_gradyan() -> None:
    # f(x, y) = x^2 + 3*y^2 → ∇f = (2x, 6y)
    f = lambda v: v[0] ** 2 + 3 * v[1] ** 2

    nokta = np.array([1.0, 2.0])
    gradyan = sayisal_gradyan(f, nokta.copy())
    beklenen = np.array([2.0, 12.0])
    assert np.allclose(gradyan, beklenen, atol=1e-6)


# -------------------------------------------------------------
# 3) Gradyan inişi — en dik iniş yönüne ufak adımlar
# -------------------------------------------------------------

def basit_gradyan_inisi() -> None:
    """
    f(x, y) = x^2 + 3*y^2 fonksiyonunun minimumu (0, 0)'dadır.
    Rastgele bir noktadan başlayıp her adımda gradyanın zıt yönünde küçük bir
    adım atarak minimuma yaklaşabilir miyiz?

    Bu, sıfırdan yazılmış en küçük "eğitim" döngüsüdür. Modül 03'ten itibaren
    her şey bunun türevi (kelime oyunu: pun intended).
    """
    f = lambda v: v[0] ** 2 + 3 * v[1] ** 2

    # Başlangıç noktası kasten "uzakta".
    x = np.array([4.0, -3.0])
    learning_rate = 0.1

    # 100 adım yeterli — quadratic'te lr=0.1 ile hızlı yakınsar.
    for _ in range(100):
        g = sayisal_gradyan(f, x.copy())
        x = x - learning_rate * g  # zıt yönde adım

    # Minimuma çok yaklaşmış olmalıyız.
    assert np.linalg.norm(x) < 1e-3, f"Beklenmedik kalıntı: {x}"


# -------------------------------------------------------------
# 4) Analitik türev ile karşılaştırma — yöntemin güvenilirliği
# -------------------------------------------------------------

def analitik_karsilastirma() -> None:
    """
    Doğrudan elle türevini bildiğimiz fonksiyonlarda sayısal türev ile
    analitik türevin yakınsadığını kontrol edelim. Bu, ileride autograd
    çıktısını da doğrularken kullanacağımız tekniğin minimal hali.
    """
    rng = np.random.default_rng(0)

    # f(x) = (x · x) / 2  → ∇f(x) = x   (sade ve hatırlanması kolay)
    f = lambda v: 0.5 * np.dot(v, v)
    x = rng.standard_normal(8)

    sayisal = sayisal_gradyan(f, x.copy())
    analitik = x.copy()

    # Sayısal türev mükemmel değildir; tolerans bırakmak gerekir.
    assert np.allclose(sayisal, analitik, atol=1e-6)


if __name__ == "__main__":
    tek_degiskenli_turev()
    cok_degiskenli_gradyan()
    basit_gradyan_inisi()
    analitik_karsilastirma()
    print("07_turev_sezgisi: tüm assert'ler geçti.")
