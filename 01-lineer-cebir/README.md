# Modül 01 — Lineer Cebir

> "`A @ x` aslında ne yapıyor?" — Bu modülün baştan sona cevapladığı soru.

Modül 00 NumPy mekaniğini verdi. Bu modül o mekaniğe **geometrik anlam** veriyor.
Bir matrisi artık "sayı tablosu" olarak değil, **uzayı eğip büken bir lineer dönüşüm**
olarak görmeye başlayacaksın. Attention'da `Q @ K.T` görüp orada ne olduğunu içten
hissetmek için gerekli alt yapı tam burada inşa ediliyor.

---

## 🎯 Bu Modülde Ne Öğreneceksin?

1. Bir matrisi **lineer dönüşüm** olarak okumak — sütunlar, standart taban vektörlerinin nereye gittiğidir.
2. Matris çarpımının neden böyle tanımlandığını — **dönüşümlerin kompozisyonu** olduğunu — geometrik olarak açıklamak.
3. **Özdeğer/özvektör**'ü "dönüşümün doğal eksenleri" olarak yorumlamak.
4. **SVD**'yi her matrisin "döndür → ölçekle → döndür" üçlü ayrışımı olarak okumak.
5. **Düşük-rank yaklaşım**'ın LLM dünyasında neden merkezi olduğunu (LoRA, PCA, embedding kompresyonu) görmek.

---

## 🧭 Bu Modülün Yol Haritası

| Dosya | Kavram | Neden Burada? |
|-------|--------|---------------|
| `01_vektorler_geometri.py` | Vektör geometrisi, projeksiyon | Tüm modülün geometrik dili burada açılır. |
| `02_matris_lineer_donusum.py` | Matris = uzayın dönüşümü | Bu modülün **en önemli** sezgisi. |
| `03_matris_carpim_kompozisyon.py` | `AB` = `A`'dan sonra `B` değil, **B'den sonra A** | `AB ≠ BA` kafa karışıklığının kaynağı ve çözümü. |
| `04_determinant_ters.py` | det = alan/hacim çarpanı, tersinirlik | Bir dönüşüm bilgi kaybediyor mu? Buradan anlaşılır. |
| `05_ozdeger_ozvektor.py` | Doğal eksenler, köşegenleştirme | "Bu matrisin gerçek yapısı ne?" sorusunun cevabı. |
| `06_svd.py` | `A = U Σ Vᵀ` — her matrisin canonical formu | ML'in en sık kullandığı tek bir lineer cebir teoremi. |
| `07_dusuk_rank_yaklasim.py` | Rank-k yaklaşım, PCA mini, **LoRA önizlemesi** | Modül 06'da göreceğin LoRA'nın hazırlığı. |
| `notebook.ipynb` | İnteraktif keşif | 2B dönüşüm görselleri, özvektör, SVD ile resim sıkıştırma. |

---

## 📐 Ön Bilgi

- **Modül 00** zorunlu: NumPy `ndarray`, broadcasting, eksenler, `@` operatörü.
- Lise düzeyi matematik (vektör toplama, basit trigonometri) yeterli.

---

## 🧪 Kütüphane Politikası

Yalnızca `numpy` + `matplotlib`. `np.linalg.eig`, `np.linalg.svd` gibi yardımcıları **serbestçe** kullanıyoruz; çünkü hedef, sıfırdan SVD yazmak değil, çıktıyı **geometrik olarak yorumlamak**. Modül 02'de PyTorch geldiğinde aynı kavramlar `torch.linalg` üzerinden tekrar gelecek.

---

## ▶️ Nasıl Çalıştırılır?

```bash
python 02_matris_lineer_donusum.py
```

Her dosyanın altındaki `if __name__ == "__main__":` bloğu `assert`'lerle bittiğinde "tüm assert'ler geçti" çıktısı vermelidir.

Notebook için:

```bash
jupyter lab notebook.ipynb
```

---

## ✅ Modül Sonu Kontrol Listesi

Bu modülü bitirdiğinde:

- [ ] Verilen bir 2×2 matrisin **yaptığı dönüşümü kafadan tarif edebilirsin** (rotasyon? ölçekleme? shear? aynalama?).
- [ ] `AB ≠ BA`'nın neden geometrik olarak doğal olduğunu açıklayabilirsin.
- [ ] Bir matrisin özdeğerlerinden, o matrisin "ana yönde ne kadar uzattığını" söyleyebilirsin.
- [ ] SVD'yi tek cümlede özetleyebilirsin: "**U döndürür, Σ ölçekler, Vᵀ yine döndürür.**"
- [ ] Bir matrisi rank-k ile yaklaşıkladığında **ne kaybettiğini Frobenius normuyla** ifade edebilirsin.
- [ ] LoRA'nın `W + AB` formunun neden **az parametreyle çok iş** çıkardığını öngörebilirsin.

---

## 🔗 Önemli Köprüler

- `04 → 05`: Determinant 0 → tersi yok → en az bir özdeğer 0.
- `05 → 06`: Simetrik matrisin özdeğer ayrışımı, SVD'nin özel halidir.
- `06 → 07`: Eckart-Young teoremi — düşük-rank yaklaşımın hatasını SVD doğrudan verir.
- `07 → Modül 04 (Attention)`: Attention skor matrisi pratikte düşük-rank yapı taşır — sezgi burada başlar.
- `07 → Modül 06 (LoRA)`: `W + αAB` formülünün kalbi bu dosya.

---

## 🔗 Sonraki Adım

[`02-pytorch-giris/`](../02-pytorch-giris/) — Tensörler, autograd, sıfırdan bir mini sinir ağı. Aynı lineer cebir kavramları bu sefer GPU'ya çıkacak ve otomatik türev kazanacak.
