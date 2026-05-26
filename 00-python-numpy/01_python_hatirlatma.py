"""
01 — Python Hatırlatma

Sonraki dosyalarda sıkça kullanacağımız Python idiyomlarını tek bir yerde topluyoruz.
Amaç: ileride bir satır kodda "bu da neydi?" diyerek durmamak.

Burada amaç Python'u sıfırdan öğretmek değil, projenin geri kalanında kullanılacak
yapıların ortak zemine çekilmesi. Bu yüzden olabildiğince *bizim* kullanacağımız
örneklere odaklanılıyor.
"""

from typing import Iterable


# -------------------------------------------------------------
# 1) Liste, sözlük, set
# -------------------------------------------------------------

def temel_koleksiyonlar() -> None:
    # Listeler sıralıdır; index ile erişim O(1)'dir.
    # NumPy'de göreceğimiz "vektör" zihinsel olarak Python listesinin
    # sabit-tipli, daha hızlı kuzeni gibi düşünülebilir.
    sayilar = [3, 1, 4, 1, 5, 9, 2, 6]

    # Sözlükler anahtar-değer eşlemesidir; arama O(1).
    # Tokenizer kelime → id eşlemesinde tam olarak bu yapıyı kullanacağız.
    kelime_to_id = {"merhaba": 0, "dünya": 1, "model": 2}

    # Set: sıralanmamış, tekrarsız. Vocab oluştururken benzersiz tokenleri toplamak için ideal.
    benzersiz_harfler = set("merhaba dünya")

    assert kelime_to_id["model"] == 2
    assert len(benzersiz_harfler) > 0
    assert sum(sayilar) == 31


# -------------------------------------------------------------
# 2) List comprehension
# -------------------------------------------------------------

def comprehension_ornekleri() -> None:
    # Comprehension, "her eleman için işle ve topla" desenini tek satırda ifade eder.
    # Veri ön-işleme adımlarında (örn. her cümleyi tokenize et) bolca kullanacağız.

    kareler = [x * x for x in range(6)]
    assert kareler == [0, 1, 4, 9, 16, 25]

    # Koşullu comprehension: yalnızca filtreyi geçenler dahil olur.
    cift_kareler = [x * x for x in range(10) if x % 2 == 0]
    assert cift_kareler == [0, 4, 16, 36, 64]

    # Sözlük comprehension'ı, id ↔ token eşlemesini tersine çevirmekte sık görülür.
    id_to_kelime = {i: k for k, i in {"a": 0, "b": 1, "c": 2}.items()}
    assert id_to_kelime[1] == "b"


# -------------------------------------------------------------
# 3) Fonksiyonlar: tip ipucu, varsayılan değer, *args, **kwargs
# -------------------------------------------------------------

def olcekli_toplam(degerler: Iterable[float], olcek: float = 1.0) -> float:
    # Tip ipuçları zorunlu değil ama okunabilirliği artırır.
    # PyTorch fonksiyonlarında shape'leri tip yorumuyla belgelemek yaygın bir alışkanlıktır.
    return olcek * sum(degerler)


def degisken_argumanli(*sayilar: float, **etiketler: str) -> dict:
    # *args konumsal argümanları tuple olarak toplar.
    # **kwargs isimli argümanları sözlük olarak toplar.
    # Eğitim döngülerinde "hyperparam sözlüğü"nü açmak için sıkça karşılaşılır.
    return {"toplam": sum(sayilar), "etiketler": etiketler}


def fonksiyon_ornekleri() -> None:
    assert olcekli_toplam([1, 2, 3]) == 6.0
    assert olcekli_toplam([1, 2, 3], olcek=2) == 12.0

    sonuc = degisken_argumanli(1, 2, 3, ad="deney", versiyon="v1")
    assert sonuc["toplam"] == 6
    assert sonuc["etiketler"]["ad"] == "deney"


# -------------------------------------------------------------
# 4) Sınıflar — minimum şekilde
# -------------------------------------------------------------

class CalisanOrtalama:
    """
    Çalışan (running) ortalama tutucu.

    Bu kalıp, eğitim sırasında kaybın hareketli ortalamasını izlemek için neredeyse
    her projede karşımıza çıkacak. Sıfırdan yazmak Python class mantığını da hatırlatır.
    """

    def __init__(self) -> None:
        # __init__ "constructor" değil; sadece nesne oluşturulduktan sonra çağrılan
        # ilk metottur. Asıl oluşturma __new__ tarafından yapılır — ama o detaya
        # bu projede ihtiyacımız yok.
        self.toplam: float = 0.0
        self.adet: int = 0

    def ekle(self, deger: float) -> None:
        self.toplam += deger
        self.adet += 1

    def ortalama(self) -> float:
        # Sıfıra bölünmeyi engellemek için sade bir koruma; adet 0 ise 0.0 döner.
        # Bu, eğitim başlamadan loss istemek gibi senaryoda boş bir hatadan iyidir.
        if self.adet == 0:
            return 0.0
        return self.toplam / self.adet


def sinif_ornegi() -> None:
    ort = CalisanOrtalama()
    for x in [2.0, 4.0, 6.0]:
        ort.ekle(x)
    assert ort.ortalama() == 4.0


# -------------------------------------------------------------
# 5) Doğrudan çalıştırma
# -------------------------------------------------------------

if __name__ == "__main__":
    temel_koleksiyonlar()
    comprehension_ornekleri()
    fonksiyon_ornekleri()
    sinif_ornegi()
    print("01_python_hatirlatma: tüm assert'ler geçti.")
