 # -*- coding: utf-8 -*-
"""
Created on Mon May 18 15:32:17 2026

@author: yz
"""

import pandas as pd
from rouge_score import rouge_scorer

#dosya yukleme
file = "hazirlik_verisi_TUMU.csv"
df = pd.read_csv(file)

# Sütunlarımızı kontrol etmek için
girdiler = df['Girdi_Belirtec_Listesi'].tolist()
referans_ozetler = df['Hedef_Ozet'].tolist()

print(f"{file} başarıyla yüklendi. Toplam {len(referans_ozetler)} örnek var.")


metric = ['rouge1', 'rouge2', 'rougeL']

#hesaplayiciyi ayaga kaldirmak icin.
scorer = rouge_scorer.RougeScorer(metric,use_stemmer=True)
#use_stemmer=True parametresi kelimelerin köklerine inerek eşleşme yakalamasını sağlar, bu da İngilizce metinlerde daha sağlıklı sonuç verir.)

# Skorları biriktirmek için boş listeler oluşturuyoruz
rouge1_f1_listesi = []
rouge2_f1_listesi = []
rougeL_f1_listesi = []

# Tüm satırları tek tek dönüyoruz
for i in range(len(referans_ozetler)):
    tahmin_ozet = girdiler[i] # Buraya gerçek projedeki model_çıktısı_listesi[i] gelecek
    gercek_ozet = referans_ozetler[i]
    
    # ROUGE skorunu hesapla
    skorlar = scorer.score(gercek_ozet, tahmin_ozet)
    
    # F1 skorlarını listelere ekle (precision veya recall da çekebilirsin)
    rouge1_f1_listesi.append(skorlar['rouge1'].fmeasure)
    rouge2_f1_listesi.append(skorlar['rouge2'].fmeasure)
    rougeL_f1_listesi.append(skorlar['rougeL'].fmeasure)

print("Tüm satırlar için ROUGE skorları tek tek hesaplandı!")
# Listelerin ortalamasını alıp 100 ile çarparak yüzdeye çeviriyoruz
ort_rouge1 = (sum(rouge1_f1_listesi) / len(rouge1_f1_listesi)) * 100
ort_rouge2 = (sum(rouge2_f1_listesi) / len(rouge2_f1_listesi)) * 100
ort_rougeL = (sum(rougeL_f1_listesi) / len(rougeL_f1_listesi)) * 100

print("\n" + "="*40)
# Makalelerdeki tablolara yazılacak nihai F1 sonuçları:
print(f" ROUGE-1 (F1): %{ort_rouge1:.2f}")
print(f" ROUGE-2 (F1): %{ort_rouge2:.2f}")
print(f" ROUGE-L (F1): %{ort_rougeL:.2f}")
print("="*40)
