import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from datasets import load_dataset
from transformers import pipeline
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

nltk.download('stopwords',quiet='true')
stop_words=set(stopwords.words('english'))

dataset=load_dataset("kmfoda/booksum")
ner=pipeline("ner",model="Davlan/distilbert-base-multilingual-cased-ner-hrl",aggregation_strategy="simple")
keybert=KeyBERT()
print("modeller yüklendi")

def chapter_onsoz_silme(metin):
    metin=metin.strip() #metnin en başındaki ve en sonundaki tüm gereksiz boşlukları, görünmez karakterleri siler
    #Python tarafından özel kaçış karakteri ( \n  \t tab) olarak değil düz metin karakteri olarak yorumlanmasını sağlar.
    eslesme=re.search(r'^["\'“]+.*?["\'”]+', metin, re.DOTALL)
    if eslesme:
        kalan_metin=metin[eslesme.end():].strip() #alıntı sonrasını aldık
        parcalar=re.split(r'\n\s*\n',kalan_metin,maxsplit=1)
        if len(parcalar==2) and len(parcalar[0].strip())<80:
            metin=parcalar[1].strip()
        else:
            metin=kalan_metin
    return metin
            
def metni_temizle(metin): 
    metin = re.sub(r'\d+', '', metin) 
    metin = metin.lower()
    metin = re.sub(r'[^\w\s]', '', metin) 
    kelimeler = metin.split() 
    temiz_kelimeler = [kelime for kelime in kelimeler if kelime not in stop_words] 
    return " ".join(temiz_kelimeler) 

def varlik_isimlerini_bul(metin):
    ner_sonuclari = ner(metin)
    varlik_isimleri = set() 
    for varlik in ner_sonuclari:
        if varlik['entity_group'] in ['PER', 'ORG', 'LOC']:
            varlik_isimleri.add(varlik['word'])
    return list(varlik_isimleri)

def anahtar_kelimeleri_bul(temiz_metin):
    anahtar_kelimeler = keybert.extract_keywords(temiz_metin, keyphrase_ngram_range=(3, 4), stop_words='english', top_n=5) 
    return [kelime[0] for kelime in anahtar_kelimeler]

def tfidf_anahtar_kelimeleri_bul(temiz_metin, kelime_sayisi=5):
    if not temiz_metin.strip():
        return []
    vectorizer = TfidfVectorizer(max_features=kelime_sayisi)
    vectorizer.fit_transform([temiz_metin])
    return list(vectorizer.get_feature_names_out())

def bas_orta_son_fragman_al(metin):
    paragraflar = re.split(r'\n\s*\n', metin)
    
    anlamli_paragraflar = []
    
    for p in paragraflar:
        p = re.sub(r'\n', ' ', p).strip() 
        
        if len(p) > 150 and not p.lower().startswith('chapter'):
            anlamli_paragraflar.append(p)
            
    if len(anlamli_paragraflar) == 0:
        return ""
    elif len(anlamli_paragraflar) == 1:
        return anlamli_paragraflar[0]
    elif len(anlamli_paragraflar) == 2:
        return f"{anlamli_paragraflar[0]}\n\n{anlamli_paragraflar[1]}"
    
    bas_paragraf = anlamli_paragraflar[0]
    orta_paragraf = anlamli_paragraflar[len(anlamli_paragraflar) // 2]
    son_paragraf = anlamli_paragraflar[-1]
    
    return f"{bas_paragraf}\n\n{orta_paragraf}\n\n{son_paragraf}"

if __name__ == "__main__":
    islenen_veriler = []
    for split_adi in dataset.keys():
        split_verisi = dataset[split_adi]
        toplam_veri_sayisi = len(split_verisi)
        
        print(f"\n---> {split_adi.upper()} verisi isleniyor ({toplam_veri_sayisi} adet) <---")
        
        for i in tqdm(range(toplam_veri_sayisi), desc=f"{split_adi} Hazırlanıyor", unit="Bölüm"):
            try:
                ornek = split_verisi[i]
                bolum_metni = ornek.get('chapter', '')
                orijinal_ozet = ornek.get('summary_text', '')
                
                if not bolum_metni or not orijinal_ozet:
                    continue
                
                bolum_metni = chapter_onsoz_silme(bolum_metni)
                    
                giris_fragmani = bas_orta_son_fragman_al(bolum_metni)
                
                islem_metni = bolum_metni[:2500] 
                
                varliklar = varlik_isimlerini_bul(islem_metni)
                temiz_metin = metni_temizle(islem_metni)
                
                anahtar_kelimeler = anahtar_kelimeleri_bul(temiz_metin)
                tfidf_kelimeler = tfidf_anahtar_kelimeleri_bul(temiz_metin)
                
                tum_anahtar_kelimeler = anahtar_kelimeler + tfidf_kelimeler
                
                belirtec_listesi = f"{giris_fragmani} \n\n {', '.join(varliklar)} \n\n {', '.join(tum_anahtar_kelimeler)}"
                
                hikaye_id = ornek.get('bid', f"bilinmeyen_id_{split_adi}_{i}")
                
                islenen_veriler.append({
                    "Hikaye_ID": hikaye_id,
                    "Girdi_Belirtec_Listesi": belirtec_listesi,
                    "Hedef_Ozet": orijinal_ozet
                })
                
            except Exception as e:
                continue
            
    df = pd.DataFrame(islenen_veriler)
    print(f"\nİşlem Tamamlandı! Toplam İşlenen Kayıt: {len(df)}")
    
    kayit_adresi = r"C:\Users\nefo.LAPTOP-S5AHVUHS\Desktop\Bitirme_dosyalar\hazirlik_verisi_TUMU.csv"
    
    df.to_csv(kayit_adresi, index=False, encoding='utf-8-sig')
    print(f"Veri seti başarıyla kaydedildi: {kayit_adresi}")