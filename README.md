# LLM Sıfırdan — Türkçe Öğrenme Rehberi

> "Anlayamadığın şeyi sıfırdan inşa et."

Bu proje, büyük dil modellerini (LLM) sıfırdan kodlayarak öğrenmek isteyenler için hazırlanmış adım adım bir Türkçe rehberdir. Hazır kütüphanelere sarılmak yerine, her bileşeni kendi ellerinizle yazarak gerçek bir anlayış inşa etmeyi hedefliyoruz.

## Felsefe

- **Anlamak > Kullanmak.** Bir şeyi yazabiliyorsan, anlıyorsundur.
- **Minimum kütüphane.** Önce NumPy, sonra PyTorch. `transformers` gibi hazır paketler yalnızca karşılaştırma için kullanılır.
- **Türkçe düşün.** Kod İngilizce, ama açıklamalar ve dokümantasyon Türkçe. Kavramları kendi dilinizde sindirin.
- **Her satırın nedeni var.** Kod yazmak değil, *neden öyle yazdığını* bilmek önemli.

---

## Yol Haritası

| Modül | Konu | Ön Koşul |
|-------|------|-----------|
| `00-python-numpy` | Python temelleri, NumPy vektör/matris işlemleri | — |
| `01-lineer-cebir` | Matris çarpımı, özdeğerler, SVD — LLM'in matematiği | 00 |
| `02-pytorch-giris` | Tensörler, autograd, basit sinir ağı | 01 |
| `03-dil-modeli-temelleri` | N-gram, bigram, karakter düzeyi dil modeli | 02 |
| `04-attention-transformer` | Self-attention, çok başlı dikkat, Transformer bloğu | 03 |
| `05-gpt-egitimi` | Sıfırdan GPT, eğitim döngüsü, kayıp izleme | 04 |
| `06-fine-tuning` | Talimat ince ayarı, veri hazırlama, LoRA temelleri | 05 |
| `07-rlhf` | Ödül modeli, PPO ile hizalama | 06 |

---

## Kurulum

Python 3.10 veya üzeri gereklidir.

```bash
# Depoyu klonla
git clone https://github.com/kullanici/llm-from-scratch-tr.git
cd llm-from-scratch-tr

# Sanal ortam oluştur
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

> **Not:** PyTorch modül 02'de eklenir. O modüle geçtiğinizde `requirements.txt` güncellenecektir.

### Jupyter Notebook başlatmak için

```bash
jupyter lab
```

---

## Her Modülün Yapısı

```
XX-modul-adi/
├── README.md          # Türkçe kavramsal anlatım
├── 01_konu.py         # Numaralı, adım adım Python dosyaları
├── 02_konu.py
└── notebook.ipynb     # İnteraktif keşif ortamı
```

---

## Katkıda Bulunmak

Katkılar memnuniyetle karşılanır. Lütfen şu kurallara uyun:

1. Her PR tek bir modül veya konuyla sınırlı olsun.
2. Kod İngilizce, yorum ve açıklamalar Türkçe.
3. Commit mesajları konvansiyonel: `feat:`, `docs:`, `fix:`, `chore:`.
4. Hazır LLM kütüphanesi kodunu doğrudan modül içeriği olarak eklemeyin; yalnızca karşılaştırma amaçlı kullanılabilir.

---

## Lisans

MIT © 2026 Ibrahim Guney

Bu projeyi özgürce kullanabilir, değiştirebilir ve dağıtabilirsiniz. Kaynak göstermek yeterlidir.
