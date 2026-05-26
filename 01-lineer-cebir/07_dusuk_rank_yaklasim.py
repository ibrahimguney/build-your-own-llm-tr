"""
07 — Düşük-Rank Yaklaşım (Truncated SVD)

SVD bir matrisi A = U Σ Vᵀ olarak ayrıştırır ve tekil değerleri büyükten küçüğe
sıralar. **Eckart-Young teoremi** der ki: ilk k bileşeni tutup geri kalanı
sıfırlamak, A'nın **Frobenius normuna göre en iyi rank-k yaklaşımıdır**.
Yani matrise rank-k bütçeyle bakmak istersek SVD'yi kesmek **kanıtlı optimumdur**.

Bu tek teoremin LLM dünyasındaki üç ayağı:
    - **LoRA**: büyük W ağırlığına düşük-rank ΔW = BA güncellemesi → r ≪ d
    - **PCA**: yüksek boyutlu veriyi ilk k bileşene project etmek
    - **Embedding sıkıştırma**: deploy sırasında d-boyutlu embedding tablosunu kısaltmak

Hepsi aynı kapıdan girer: Eckart-Young.
"""

import numpy as np


# -------------------------------------------------------------
# 1) Truncated SVD'yi elle inşa
# -------------------------------------------------------------

def truncated_svd(A: np.ndarray, k: int) -> np.ndarray:
    """A matrisinin en iyi rank-k yaklaşımını döndürür."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    # İlk k tekil değer + ilgili vektörler. Tamamını tutup küçükleri sıfırlamak
    # yerine doğrudan kesiyoruz — hem hafıza hem hesap tasarrufu.
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]


def truncated_svd_testi() -> None:
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 4))

    # k = min(m, n) → tam matris; hata tam sıfır olmalı.
    A_tam = truncated_svd(A, k=4)
    assert np.allclose(A_tam, A, atol=1e-10)

    # k < tam → yaklaşımın rank'i tam olarak k.
    A_2 = truncated_svd(A, k=2)
    assert np.linalg.matrix_rank(A_2) == 2


# -------------------------------------------------------------
# 2) Eckart-Young hatası: ‖A − A_k‖_F² = Σ_{i>k} σ_i²
# -------------------------------------------------------------

def eckart_young_hatasi() -> None:
    """
    Frobenius normu kare = tüm elemanlarının karelerinin toplamı. SVD'nin
    güzelliği: aynı miktar tekil değerlerin karelerinin toplamına eşittir.
        ‖A‖_F² = Σ σ_i²
    Bu yüzden ilk k bileşeni tutup geri kalanı atınca kalan hata tam olarak
    Σ_{i>k} σ_i² olur. Yani hatayı **hesaplamak için ek iş yapmaya gerek yok**;
    SVD bize hatayı doğrudan veriyor.
    """
    rng = np.random.default_rng(1)
    A = rng.standard_normal((8, 5))

    _, s, _ = np.linalg.svd(A, full_matrices=False)

    k = 2
    A_k = truncated_svd(A, k)
    gercek_hata_kare = np.linalg.norm(A - A_k, ord="fro") ** 2
    teorik_hata_kare = float(np.sum(s[k:] ** 2))

    assert np.isclose(gercek_hata_kare, teorik_hata_kare, atol=1e-10)


# -------------------------------------------------------------
# 3) Açıklanan varyans oranı — kümülatif enerji
# -------------------------------------------------------------

def aciklanan_varyans() -> None:
    """
    "İlk k bileşen matrisin ne kadarını açıklıyor?" sorusunun cevabı:
        oran(k) = (Σ_{i≤k} σ_i²) / (Σ_i σ_i²)

    Bu oran 0.9, 0.95 gibi eşiklere ne kadar hızlı çıktığına bakarak matrisin
    "kaç bileşen yeter" sorusunu pratik olarak cevaplarız. **Düşük efektif
    rank** taşıyan matrislerde oran çok hızlı 1'e yaklaşır — bu, doğal
    görüntülerin ve attention skor matrislerinin ortak özelliğidir.
    """
    # Bilerek rank-3 yapı inşa edelim: (20×3) @ (3×10) = (20×10), rank ≤ 3.
    rng = np.random.default_rng(2)
    U_gizli = rng.standard_normal((20, 3))
    V_gizli = rng.standard_normal((3, 10))
    A = U_gizli @ V_gizli  # rank tam 3 (gürültü yok).

    _, s, _ = np.linalg.svd(A, full_matrices=False)

    toplam_enerji = float(np.sum(s ** 2))
    kumulatif = np.cumsum(s ** 2) / toplam_enerji

    # İlk 3 bileşen, enerjinin neredeyse tamamını taşır.
    assert kumulatif[2] > 0.9999
    # 4. bileşen ihmal edilebilir düzeyde — gerçekten rank 3.
    assert (kumulatif[3] - kumulatif[2]) < 1e-10


# -------------------------------------------------------------
# 4) Görüntü sıkıştırma — sayısal kanıt (görselleştirme notebook'ta)
# -------------------------------------------------------------

def gorsel_uret_sentetik(N: int = 64) -> np.ndarray:
    """
    Test için sentetik bir 'gri-skala görüntü'. Pürüzsüz dalgalanma (düşük
    frekanslar) + lokal bir blok (keskin kenar) + küçük bir gürültü.

    Bu üçlü, doğal görüntülerin önemli bir özelliğini taklit eder:
        - Ana yapı düşük rank'lıdır (birkaç σ büyük),
        - Ama gerçek matris tam ranklıdır (sensör gürültüsü → tüm σ'lar > 0).
    Bu yüzden "düşük efektif rank" deriz, "düşük rank" değil.
    """
    x = np.linspace(0, 4 * np.pi, N)
    y = np.linspace(0, 4 * np.pi, N)
    X, Y = np.meshgrid(x, y)
    img = np.sin(X) * np.cos(Y) + 0.3 * np.sin(2 * X + Y)
    img[10:30, 10:30] += 1.5  # köşede keskin kontrastlı bir kare
    # Görüntüye küçük bir gürültü ekle — gerçek dünya kameralarında olduğu gibi.
    # Test tekrarlanabilir olsun diye sabit tohum.
    rng = np.random.default_rng(42)
    img += 0.05 * rng.standard_normal(img.shape)
    return img


def gorsel_sikistirma() -> None:
    """
    Aynı görüntüyü k=1, 2, 4, 8, 16 ile yaklaşıkla; hata **monoton azalmalı**.
    Bu, "k arttıkça yaklaşım iyileşir" sezgisinin sayısal kanıtıdır ve aynı
    zamanda Eckart-Young'un pratik sonucu.
    """
    img = gorsel_uret_sentetik()
    hatalar = []
    for k in (1, 2, 4, 8, 16):
        img_k = truncated_svd(img, k)
        hata = float(np.linalg.norm(img - img_k, ord="fro"))
        hatalar.append(hata)

    # Hatalar monoton azalmalı (eşitlik kabul; ama 64x64 görüntüde strict azalma beklenir).
    assert all(hatalar[i] >= hatalar[i + 1] for i in range(len(hatalar) - 1))


# -------------------------------------------------------------
# 5) Parametre tasarrufu hesabı
# -------------------------------------------------------------

def parametre_tasarrufu() -> None:
    """
    m × n matrisi tam tutmak: m·n parametre.
    Rank-k yaklaşımı için: U (m×k) + s (k) + Vᵀ (k×n) → k·(m + n + 1) parametre.

    Tasarruf olur, eğer:
        k·(m + n + 1) < m·n  ⇔  k < m·n / (m + n + 1)

    Tipik LLM senaryosu: d = 4096, k = 8 (LoRA varsayılan rank) →
    tasarruf yaklaşık 256× kadar olur. Eğitim sırasında tek başına bu fark
    büyük modellerin tek GPU'ya sığmasını sağlar.
    """
    m, n = 4096, 4096
    k = 8

    tam = m * n
    dusuk_rank = k * (m + n + 1)

    oran = tam / dusuk_rank
    # ~250× büyüklük civarında bir tasarruf bekliyoruz.
    assert 250 < oran < 300


# -------------------------------------------------------------
# 6) LoRA önizlemesi — W + α·B·A
# -------------------------------------------------------------

def lora_onizlemesi() -> None:
    """
    LoRA fine-tune'da bir lineer katmanın tam W ağırlığını yeniden eğitmek
    yerine **küçük bir bozulma** öğrenir:

        W_eff = W + α · B · A

    Burada B: (d × r), A: (r × d), r ≪ d. Yani fine-tune sırasında yalnızca
    2·d·r parametre eğitilir.

    Buradaki kritik gözlem: **B·A** her zaman rank ≤ r'dir. Yani bu güncelleme
    matematiksel olarak rank-r bir bozulmadır ve doğrudan Eckart-Young
    uzayında yaşar — sıfırdan eğitilebilen en verimli bozulma şekli.
    """
    rng = np.random.default_rng(3)
    d, r = 64, 4
    W = rng.standard_normal((d, d))
    B = rng.standard_normal((d, r))
    A = rng.standard_normal((r, d))
    alpha = 0.1

    delta_W = alpha * (B @ A)
    W_eff = W + delta_W

    # B·A'nın rank'i r'yi aşamaz; çoğunlukla tam r olur.
    assert np.linalg.matrix_rank(delta_W) <= r

    # Eğitilebilir parametre sayısı:
    #   Tam fine-tune: d·d = 4096
    #   LoRA:          2·d·r = 512   → bu küçük örnekte bile ~8× tasarruf.
    tam_ft = d * d
    lora_ft = 2 * d * r
    assert lora_ft < tam_ft / 4

    # W_eff hala (d, d) bir matris — kullanım anında değil eğitim anında tasarruf var.
    assert W_eff.shape == (d, d)


# -------------------------------------------------------------
# 7) Mini PCA — varyansa göre sıralı bileşenler
# -------------------------------------------------------------

def mini_pca() -> None:
    """
    PCA: veriyi merkezle, SVD al, ilk k bileşene project et. İlk bileşen
    en fazla varyansı yakalayan yöndür (= Vᵀ'nin 1. satırı).

    LLM dünyasında PCA doğrudan model parçası değildir; ama "boyut indirgeme",
    "embedding sıkıştırma", "PCA on activations for interpretability" gibi
    pratik kullanımlar tam buradaki sezgiye dayanır.
    """
    rng = np.random.default_rng(4)
    # 3D veri, ama gerçekte 2D bir düzlemde + biraz gürültü yaşıyor.
    N = 200
    duzlem_yonu = np.array([[1.0, 0.5, 0.2],
                            [0.2, -0.3, 0.8]])
    coefs = rng.standard_normal((N, 2))
    veri = coefs @ duzlem_yonu + 0.05 * rng.standard_normal((N, 3))

    # PCA için veriyi merkezliyoruz (sütun ortalamalarını sıfır yapıyoruz).
    merkezli = veri - veri.mean(axis=0, keepdims=True)

    # SVD ile PCA: bileşenler Vᵀ'nin satırlarıdır.
    _, s, Vt = np.linalg.svd(merkezli, full_matrices=False)

    # İlk 2 bileşene project et.
    bilesenler = Vt[:2, :]                    # (2, 3)
    proj = merkezli @ bilesenler.T            # (N, 2)

    # Varyans oranı: ilk 2 bileşen toplamın %95+'ini açıklamalı (zaten 2D bir düzlem).
    varyans_orani = float(np.sum(s[:2] ** 2) / np.sum(s ** 2))
    assert varyans_orani > 0.95

    # Projeksiyon 2D'ye düştü.
    assert proj.shape == (N, 2)


if __name__ == "__main__":
    truncated_svd_testi()
    eckart_young_hatasi()
    aciklanan_varyans()
    gorsel_sikistirma()
    parametre_tasarrufu()
    lora_onizlemesi()
    mini_pca()
    print("07_dusuk_rank_yaklasim: tüm assert'ler geçti.")
