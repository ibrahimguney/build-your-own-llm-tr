# CLAUDE.md — Proje Çalışma Kuralları

Bu dosya, Claude Code ile bu projede çalışırken uyulması gereken kuralları tanımlar.

## Dil Kuralı

- **Kod:** İngilizce (değişken, fonksiyon, sınıf, dosya adları)
- **Yorumlar ve docstring'ler:** Türkçe
- **Modül README.md dosyaları:** Türkçe
- **Commit mesajları:** İngilizce (konvansiyonel format)

Örnek:

```python
def compute_attention_weights(query, key):
    # Sorgu ve anahtar vektörlerinin nokta çarpımını ölçekleyerek dikkat ağırlıklarını hesapla.
    # Boyut kökü ile bölme, gradyanları kararlı tutar (Vaswani et al., 2017).
    scale = query.shape[-1] ** 0.5
    return (query @ key.T) / scale
```

## Modül Yapısı

Her modül şu yapıyı izler:

```
XX-modul-adi/
├── README.md          # Türkçe kavramsal anlatım — sezgi, matematik, bağlam
├── 01_konu.py         # Numaralı dosyalar, her biri tek bir kavramı kapsar
├── 02_konu.py
└── notebook.ipynb     # İnteraktif keşif; grafikler ve ara çıktılar burada
```

- `.py` dosyaları numaralı ve bağımsız çalışabilir olmalı.
- Notebook'lar keşif içindir; production kodu `.py` dosyalarında yaşar.
- Her modülde bir `README.md` zorunludur; kavramsal arka planı, hedefleri ve beklenen çıktıyı açıklar.

## Kütüphane Politikası

| Modül | İzin verilen |
|-------|-------------|
| 00–01 | Yalnızca `numpy`, `matplotlib` |
| 02+   | `numpy`, `matplotlib`, `torch` |
| Herhangi | `transformers`, `tokenizers` vb. **yalnızca karşılaştırma hücrelerinde** |

Hazır kütüphane implementasyonları doğrudan modül kodu olarak yazılamaz. Yalnızca "sıfırdan yazdığımız" ile "kütüphanenin çıktısı" karşılaştırması için kullanılabilir.

## Yorum Felsefesi

Her fonksiyonda **neden** öyle yapıldığı açıklanmalı:

- Matematiksel motivasyon (formül referansı, sayısal kararlılık nedeni vb.)
- Alternatif yaklaşım varsa neden bu seçildi
- Dikkat edilmesi gereken kenar durum

Gereksiz yorum yazma (kodun ne yaptığını tekrar etme). Yorumlar okuyucunun "neden?" sorusuna cevap vermelidir.

## Commit Konvansiyonu

```
feat: add scaled dot-product attention implementation
docs: translate module 03 README to Turkish
fix: correct matrix dimension mismatch in softmax
chore: add torch to requirements for module 02
```

Etiketler: `feat`, `docs`, `fix`, `refactor`, `chore`, `test`

## Görev Akışı

1. Yeni bir kavram eklerken önce modülün `README.md`'sine kavramsal açıklamayı yaz.
2. Ardından numaralı `.py` dosyasını oluştur.
3. Notebook'ta çalıştırılabilir örnek ekle.
4. Gerekirse `requirements.txt`'i güncelle ve commit mesajına not düş.

## Genel Kurallar

- Fonksiyonları mümkün olduğunca küçük ve tek sorumlu tut.
- Her `.py` dosyası `if __name__ == "__main__":` bloğuyla doğrudan çalıştırılabilir olmalı.
- Test kodu ayrı bir dosyada değil, modülün kendi `if __name__` bloğunda basit assert'lerle yapılabilir.
- `data/` klasörü `.gitignore`'da; büyük veri dosyalarını repoya commit etme.
