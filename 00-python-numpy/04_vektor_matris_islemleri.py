"""
04 — Vektör ve Matris İşlemleri

Buradaki tek bir karışıklık (element-wise mı, matmul mı?) ileride saatler kaybettirir.
Bu yüzden iki dünyayı net ayırıyoruz:

    *      → element-wise (Hadamard) çarpım — şekiller broadcast'lenebilir olmalı.
    @      → matris çarpımı (matmul / dot).
    np.dot → 1D ile 1D'de iç çarpım, 2D ile 2D'de matmul. Karışıklık kaynağı
             olabildiği için bu projede *vektör iç çarpımı için kasten kullanılıyor*,
             matrisler için her zaman `@` tercih ediyoruz.

Attention formülünün kalbi:
    Skor = Q @ K.T / sqrt(d_k)

burada Q ve K matris, `@` matris çarpımı, `K.T` transpoz, `/ sqrt(d_k)` ise
skalar broadcasting. Bu dosyayı bitirdiğinde formülün şeklinin neden bu olduğunu
göreceksin.
"""

import numpy as np


# -------------------------------------------------------------
# 1) Element-wise vs matris çarpımı — iki ayrı evren
# -------------------------------------------------------------

def element_wise_vs_matmul() -> None:
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])

    # Element-wise: A[i,j] * B[i,j].
    hadamard = A * B
    assert np.array_equal(hadamard, np.array([[5, 12], [21, 32]]))

    # Matris çarpımı: sonuç[i,j] = Σ_k A[i,k] * B[k,j].
    # Şekil kuralı: (m, n) @ (n, p) → (m, p).
    matmul = A @ B
    assert np.array_equal(matmul, np.array([[19, 22], [43, 50]]))


# -------------------------------------------------------------
# 2) Vektörlerle iç çarpım, dış çarpım
# -------------------------------------------------------------

def ic_carpim_dis_carpim() -> None:
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([4.0, -5.0, 6.0])

    # İç çarpım (dot product): tek bir skaler.
    # Geometrik anlamı: |u| * |v| * cos(theta). İki vektör arasındaki "hizalanma"
    # ölçüsüdür ve attention'da Q ile K vektörlerinin benzerliğini ölçer.
    ic = np.dot(u, v)
    # 1*4 + 2*(-5) + 3*6 = 4 - 10 + 18 = 12
    assert np.isclose(ic, 12.0)

    # Dış çarpım (outer product): vektör × vektör → matris.
    # outer(u, v)[i, j] = u[i] * v[j]
    # İleri seviyede LoRA'da düşük-rank güncelleme tam olarak bir dış çarpımdır:
    # ΔW ≈ a · b^T  (Modül 06'da göreceğiz).
    dis = np.outer(u, v)
    assert dis.shape == (3, 3)
    assert np.isclose(dis[0, 0], 4.0)
    assert np.isclose(dis[2, 2], 18.0)


# -------------------------------------------------------------
# 3) Transpoz — eksen takası
# -------------------------------------------------------------

def transpoz_ornegi() -> None:
    # (m, n) bir matrisin transpozu (n, m)'dir. M[i, j] ↔ M.T[j, i].
    # Attention'da `K.T` kullanmamızın nedeni budur: K matrisi (T, d_k) iken,
    # `Q @ K.T` çarpımı (T, T) skor matrisi üretir. Yani "her token'in her token ile benzerliği".
    M = np.arange(6).reshape(2, 3)  # shape (2, 3)
    Mt = M.T                        # shape (3, 2)
    assert M.shape == (2, 3)
    assert Mt.shape == (3, 2)
    assert M[0, 1] == Mt[1, 0]


# -------------------------------------------------------------
# 4) Norm — bir vektörün uzunluğu
# -------------------------------------------------------------

def norm_ornegi() -> None:
    # L2 norm: sqrt(Σ x_i^2). Vektörün Öklid uzunluğu.
    # `np.linalg.norm` varsayılan olarak L2 normunu hesaplar.
    v = np.array([3.0, 4.0])
    assert np.isclose(np.linalg.norm(v), 5.0)  # Pisagor üçgeni 3-4-5.

    # L1 norm: Σ |x_i|. Sparsity üretmesiyle bilinir.
    # ord=1 parametresi ile çağrılır; LLM eğitiminde nadiren ama yine de görülür.
    assert np.isclose(np.linalg.norm(v, ord=1), 7.0)


# -------------------------------------------------------------
# 5) Kosinüs benzerliği — embedding'lerin dilini konuşmak
# -------------------------------------------------------------

def kosinus_benzerligi(u: np.ndarray, v: np.ndarray) -> float:
    """
    İki vektör arasındaki kosinüs benzerliği.

    cos(theta) = (u · v) / (|u| * |v|)

    Sonuç [-1, 1] aralığındadır:
      +1 → aynı yönde,
       0 → dik (alakasız),
      -1 → zıt yönde.

    Embedding'lerin yön bilgisi büyüklükten önemli olduğu için bu metrik
    LLM dünyasında baskındır. Ayrıca normlar farklı olduğunda da adil bir karşılaştırma sağlar.
    """
    # Sıfır vektörü kontrolü: normlar 0 ise tanımsız bir sonuç olur.
    # Test ortamında bu durumla karşılaşmıyoruz ama gerçek veride
    # padding token embedding'i sıfır olabilir; o yüzden eklemek mantıklı.
    eps = 1e-12
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    return float(np.dot(u, v) / (norm_u * norm_v + eps))


def kosinus_testi() -> None:
    # Aynı yön → 1.
    assert np.isclose(kosinus_benzerligi(np.array([1, 0]), np.array([5, 0])), 1.0)
    # Dik → 0.
    assert np.isclose(kosinus_benzerligi(np.array([1, 0]), np.array([0, 1])), 0.0)
    # Zıt → -1.
    assert np.isclose(kosinus_benzerligi(np.array([1, 0]), np.array([-2, 0])), -1.0)


# -------------------------------------------------------------
# 6) Mini attention skor matrisi — bir önizleme
# -------------------------------------------------------------

def mini_attention_skoru() -> None:
    """
    Modül 04'te detaylanacak ama buraya bir önizleme koyalım: 3 token, her biri
    4 boyutlu vektör. Attention skor matrisi (3, 3) olur.
    """
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((3, 4))
    K = rng.standard_normal((3, 4))

    d_k = Q.shape[-1]
    # Q @ K.T → (3, 3) skor matrisi.
    # /sqrt(d_k) → büyük d_k'lerde softmax'in saturate olmasını engeller (Vaswani et al., 2017).
    skor = (Q @ K.T) / np.sqrt(d_k)

    assert skor.shape == (3, 3)
    # Köşegen olması gerekmez; ama sayısal olarak makul aralıkta olmalı.
    assert np.all(np.isfinite(skor))


if __name__ == "__main__":
    element_wise_vs_matmul()
    ic_carpim_dis_carpim()
    transpoz_ornegi()
    norm_ornegi()
    kosinus_testi()
    mini_attention_skoru()
    print("04_vektor_matris_islemleri: tüm assert'ler geçti.")
