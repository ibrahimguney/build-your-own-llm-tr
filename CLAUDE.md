# CLAUDE.md — Proje Çalışma Kuralları

Bu dosya, Claude Code ile bu projede çalışırken uyulması gereken kuralları tanımlar.

## Dil Kuralı

- **Dosya ve klasör adları:** İngilizce veya Türkçe; **ASCII olmalı** (Türkçe karakter yok).
- **Fonksiyon, değişken, sınıf adları:** Türkçe (sezgi için) veya İngilizce (genel API için) serbest; **ASCII olmalı**. Türkçe `ı/ğ/ş/ç/ö/ü` karakterleri yerine ASCII karşılıkları kullanılır: `kimlik_karti` (`kimlik_kartı` değil), `markov_duragan_dagilim` (`markov_durağan_dağılım` değil).
- **Yorumlar ve docstring'ler:** Türkçe; her tür karakter serbest.
- **Modül README.md dosyaları:** Türkçe.
- **Commit mesajları:** İngilizce (konvansiyonel format).

Örnek (Türkçe pratiği — projenin baskın stili):

```python
def kosinus_benzerligi(u, v):
    # İki vektör arasındaki yön benzerliğini ölçer.
    # Embedding'lerde büyüklük değil yön önemli olduğu için tercih edilir.
    return (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))
```

İngilizce identifier ile yazmak da geçerli — özellikle "framework arayüzü" gibi
duran fonksiyonlar için (`compute_attention_weights`, `scaled_dot_product`).
İki stil aynı dosyada karışabilir.

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
