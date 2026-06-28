import re
import torch
import nltk
import gradio as gr
from nltk.corpus import stopwords
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
import fitz  # PyMuPDF
import docx

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# ============================================================
# MODELLERİ YÜKLE
# ============================================================
print("Modeller yükleniyor...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ner = pipeline(
    "ner",
    model="Davlan/distilbert-base-multilingual-cased-ner-hrl",
    aggregation_strategy="simple",
    device=0 if torch.cuda.is_available() else -1,
)
keybert = KeyBERT()

# Kendi eğittiğimiz modelin klasör yolu
tokenizer  = AutoTokenizer.from_pretrained("bart_hikaye_model", use_fast=True)
bart_model = AutoModelForSeq2SeqLM.from_pretrained("bart_hikaye_model").to(device)
bart_model.eval()
print(f"Tüm modeller hazır | Cihaz: {device}")

MAX_INPUT  = 1024
MAX_TARGET = 512

# ============================================================
# DOSYA OKUMA FONKSİYONLARI
# ============================================================
def pdf_oku(yol):
    try:
        doc   = fitz.open(yol)
        metin = "".join(sayfa.get_text("text") for sayfa in doc)
        doc.close()
        return metin.strip()
    except Exception as e:
        return f"PDF okunamadı: {e}"

def word_oku(yol):
    try:
        doc = docx.Document(yol)
        return "\n\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        return f"Word okunamadı: {e}"

def dosya_isle(dosya):
    """
    UploadButton ile dosya yüklendiğinde çağrılır.
    Dosyadan metni okur ve Textbox'a yazar.
    """
    if dosya is None:
        return ""
    
    yol = dosya.name if hasattr(dosya, 'name') else dosya
    
    if yol.lower().endswith(".pdf"):
        return pdf_oku(yol)
    elif yol.lower().endswith((".docx", ".doc")):
        return word_oku(yol)
    else:
        return "Sadece PDF ve DOCX desteklenir!"

# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================
def chapter_onsoz_silme(metin):
    metin = metin.strip()
    eslesme = re.search(r'^["\'"]+.*?["\'"]+', metin, re.DOTALL)
    if eslesme:
        kalan_metin = metin[eslesme.end():].strip()
        parcalar = re.split(r'\n\s*\n', kalan_metin, maxsplit=1)
        if len(parcalar) == 2 and len(parcalar[0].strip()) < 80:
            metin = parcalar[1].strip()
        else:
            metin = kalan_metin
    return metin

def metni_temizle(metin):
    metin = re.sub(r'\d+', '', metin)
    metin = metin.lower()
    metin = re.sub(r'[^\w\s]', '', metin)
    kelimeler = metin.split()
    return " ".join(k for k in kelimeler if k not in stop_words)

def varlik_isimlerini_bul(metin):
    ner_sonuclari = ner(metin)
    varlik_isimleri = set()
    for v in ner_sonuclari:
        if v['entity_group'] in ['PER', 'ORG', 'LOC']:
            varlik_isimleri.add(v['word'])
    return list(varlik_isimleri)

def anahtar_kelimeleri_bul(temiz_metin):
    sonuc = keybert.extract_keywords(
        temiz_metin, keyphrase_ngram_range=(3, 4),
        stop_words='english', top_n=5,
    )
    return [k[0] for k in sonuc]

def tfidf_anahtar_kelimeleri_bul(temiz_metin, kelime_sayisi=5):
    if not temiz_metin.strip():
        return []
    v = TfidfVectorizer(max_features=kelime_sayisi)
    v.fit_transform([temiz_metin])
    return list(v.get_feature_names_out())

def bas_orta_son_fragman_al(metin):
    paragraflar = re.split(r'\n\s*\n', metin)
    anlamli = []
    for p in paragraflar:
        p = re.sub(r'\n', ' ', p).strip()
        if len(p) > 150 and not p.lower().startswith('chapter'):
            anlamli.append(p)
    if len(anlamli) == 0:   return ""
    if len(anlamli) == 1:   return anlamli[0]
    if len(anlamli) == 2:   return f"{anlamli[0]}\n\n{anlamli[1]}"
    return f"{anlamli[0]}\n\n{anlamli[len(anlamli)//2]}\n\n{anlamli[-1]}"

# ============================================================
# ANA PIPELINE
# ============================================================
def pipeline_calistir(ham_metin, min_uzunluk, num_beams, length_penalty):
    if not ham_metin.strip():
        return "", "", "", "", "Lütfen metin girin veya dosya yükleyin."

    temiz_ham      = chapter_onsoz_silme(ham_metin)
    giris_fragmani = bas_orta_son_fragman_al(temiz_ham)
    islem_metni    = temiz_ham[:2500]
    varliklar      = varlik_isimlerini_bul(islem_metni)
    temiz_metin    = metni_temizle(islem_metni)
    anahtar        = anahtar_kelimeleri_bul(temiz_metin)
    tfidf          = tfidf_anahtar_kelimeleri_bul(temiz_metin)
    tum_kelimeler  = anahtar + tfidf

    belirtec_listesi = (
        "Reconstruct the chapter's storyline into a cohesive narrative "
        "by connecting the provided excerpts, and utilizing the key "
        "characters and keywords:\n\n"
        f"{giris_fragmani} \n\n "
        f"{', '.join(varliklar)} \n\n "
        f"{', '.join(tum_kelimeler)}"
    )

    inputs = tokenizer(
        belirtec_listesi,
        max_length=MAX_INPUT,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            output_ids = bart_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=MAX_TARGET,
                min_length=int(min_uzunluk),
                num_beams=int(num_beams),
                length_penalty=float(length_penalty),
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

    ozet       = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    varlik_str = ", ".join(varliklar)     if varliklar     else "Bulunamadı"
    kelime_str = ", ".join(tum_kelimeler) if tum_kelimeler else "Bulunamadı"
    durum      = f"Tamamlandı — {len(tokenizer.encode(ozet))} token"

    return ozet, varlik_str, kelime_str, giris_fragmani, durum

# ============================================================
# ARAYÜZ 
# ============================================================
with gr.Blocks(title="Hikaye Özetleyici", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 📖 Hikaye Bölümü Özetleyici")
    gr.Markdown(
        "Metin kutusuna yaz **veya** aşağıdaki **➕ Dosya Yükle** butonuna tıklayarak PDF/DOCX dosyanı seç. "
        "NER, KeyBERT ve TF-IDF otomatik çalışır."
    )

    with gr.Row():
        with gr.Column(scale=2):
            
            ham_metin_input = gr.Textbox(
                label="Ham Hikaye Metni",
                placeholder="Kitap bölümünü buraya yaz veya alttaki butondan PDF/DOCX yükle...",
                lines=12,
            )
            
            # ➕ Tıklandığında direkt pencere açan buton
            dosya_yukle_btn = gr.UploadButton(
                "➕ Dosya Yükle (PDF / DOCX)", 
                file_types=[".pdf", ".docx", ".doc"], 
                variant="secondary"
            )

            with gr.Accordion("⚙️ Gelişmiş Ayarlar", open=False):
                min_uzunluk    = gr.Slider(20,  150, value=40,  step=5,   label="Min özet uzunluğu (token)")
                num_beams      = gr.Slider(1,   8,   value=4,   step=1,   label="Beam sayısı")
                length_penalty = gr.Slider(0.5, 3.0, value=2.0, step=0.1, label="Length penalty")

            uret_btn = gr.Button("Özet Üret", variant="primary")

        with gr.Column(scale=2):
            ozet_cikti  = gr.Textbox(label="📝 Üretilen Özet", lines=8,  interactive=False)
            durum_cikti = gr.Textbox(label="Durum",            lines=1,  interactive=False)

            with gr.Accordion("🔍 Pipeline Detayları", open=False):
                # max_lines ile kaydırma çubuğu (scrollbar) özelliği eklendi!
                varlik_cikti  = gr.Textbox(label="Varlık İsimleri (NER)",              lines=2, max_lines=4, interactive=False)
                kelime_cikti  = gr.Textbox(label="Anahtar Kelimeler (KeyBERT+TF-IDF)", lines=2, max_lines=4, interactive=False)
                fragman_cikti = gr.Textbox(label="Girdi Paragrafı",                    lines=4, max_lines=6, interactive=False)

    # 1. Dosya yüklenince çalışan fonksiyon
    dosya_yukle_btn.upload(
        fn=dosya_isle,
        inputs=dosya_yukle_btn,
        outputs=ham_metin_input,
    )

    # 2. Özet Üret butonuna basılınca çalışan fonksiyon
    uret_btn.click(
        fn=pipeline_calistir,
        inputs=[ham_metin_input, min_uzunluk, num_beams, length_penalty],
        outputs=[ozet_cikti, varlik_cikti, kelime_cikti, fragman_cikti, durum_cikti],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )