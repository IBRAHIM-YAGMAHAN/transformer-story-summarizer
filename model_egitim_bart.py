import os
import csv
import gc
import time
import math
import random
from contextlib import nullcontext

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup,
)


#degistirdiklerim length_penalty=2'di onu degistiredim bide MAX_TARGET=216'di#
# sihirli_girdi = f"Reconstruct the chapter's storyline into a cohesive narrative by connecting the provided excerpts, and utilizing the key characters and keywords:\n\n{girdi}"


# ============================================================
# 1) AYARLAR
# ============================================================

CSV_PATH = "hazirlik_verisi_TUMU.csv" 


BASE_MODEL_NAME = "facebook/bart-large-cnn"

# Epoch sonunda en iyi model buraya kaydedilir.
BEST_MODEL_DIR = "bart_hikaye_model"

# Eğitim log dosyası
LOG_PATH = "egitim_log.txt"

SEED = 42


MAX_INPUT = 1024
MAX_TARGET = 512 #256di sadece bunu degistirdik

BATCH_SIZE = 1
GRAD_ACCUM = 16

EPOCHS = 10
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 200

LOG_EVERY = 50
EARLY_STOPPING_PATIENCE = 3

USE_GRADIENT_CHECKPOINTING = True

RESUME_FROM_BEST_MODEL_IF_EXISTS = True
SAVE_TRAINING_STATE = False
LAST_STATE_DIR = "bart_hikaye_training_state"


# ============================================================
# 2) YARDIMCI FONKSİYONLAR
# ============================================================

def log(msg: str):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gpu_info():
    if not torch.cuda.is_available():
        return "GPU yok"

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    name = torch.cuda.get_device_name(0)

    return f"{name} | allocated={allocated:.2f} GB | reserved={reserved:.2f} GB"


def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 3) VERİYİ OKUMA 
# ============================================================

def read_dataset(csv_path: str):
    rows = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Sadece yeni oluşturduğumuz iki kolonu okuyoruz, ID dahil diğer her şeyi atlıyoruz
            girdi = (row.get("Girdi_Belirtec_Listesi") or "").strip()
            hedef = (row.get("Hedef_Ozet") or "").strip()

            if not girdi or not hedef:
                continue
                
    
            sihirli_girdi = f"Reconstruct the chapter's storyline into a cohesive narrative by connecting the provided excerpts, and utilizing the key characters and keywords:\n\n{girdi}"

            rows.append({
                "input": sihirli_girdi,
                "target": hedef,
            })

    if len(rows) < 10:
        raise ValueError("Çok az veri okundu. CSV'yi kontrol et.")
        
    return rows


# ============================================================
# 4) DATASET
# ============================================================

class HikayeDataset(Dataset):
    def __init__(self, veriler, tokenizer, max_input_len, max_target_len):
        self.veriler = veriler
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.veriler)

    def __getitem__(self, idx):
        ornek = self.veriler[idx]

        model_inputs = self.tokenizer(
            ornek["input"],
            max_length=self.max_input_len,
            truncation=True,
            padding=False,
        )

        labels = self.tokenizer(
            text_target=ornek["target"],
            max_length=self.max_target_len,
            truncation=True,
            padding=False,
        )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs


# ============================================================
# 5) VALIDASYON
# ============================================================

def evaluate(model, val_loader, device, use_fp16):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    autocast_ctx = torch.cuda.amp.autocast if use_fp16 else nullcontext

    with torch.no_grad():
        for batch in val_loader:
            batch = {
                k: v.to(device, non_blocking=True)
                for k, v in batch.items()
            }

            with autocast_ctx():
                outputs = model(**batch)
                loss = outputs.loss

            total_loss += loss.item()
            total_batches += 1

    if total_batches == 0:
        return float("inf")

    return total_loss / total_batches


# ============================================================
# 6) ÖRNEK TAHMİN 
# ============================================================

def show_sample_prediction(model, tokenizer, sample, device):
    model.eval()

    inputs = tokenizer(
        sample["input"],
        max_length=MAX_INPUT,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                output_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=MAX_TARGET,
                    min_length=40,
                    num_beams=4,
                    length_penalty=1.57, # DİKKAT: Modelin uzun ve doyurucu cümleler kurması için 2.0 yapıldı.
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
        else:
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=MAX_TARGET,
                min_length=40,
                num_beams=4,
                length_penalty=1.57, # DİKKAT: Burada da 2.0 yapıldı.
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

    model_ozeti = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    log("\n=== ÖRNEK TAHMİN ===")
    log("\n[GİRDİ ]")
    log(sample["input"][:])

    log("\n[MODELİN ÜRETTİĞİ ÖZET]")
    log(model_ozeti)

    log("\n[GERÇEK ÖZET ]")
    log(sample["target"][:])


# ============================================================
# 7) ANA EĞİTİM FONKSİYONU
# ============================================================

def main():
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("")

    set_seed(SEED)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log("=" * 70)
    log("BART HİKÂYE ÖZETLEME EĞİTİMİ (YENİ NESİL)")
    log("=" * 70)
    log(f"Cihaz: {device}")

    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"CUDA version: {torch.version.cuda}")

    # ------------------------------------------------------------
    # Veri
    # ------------------------------------------------------------
    rows = read_dataset(CSV_PATH)

    train_val, test = train_test_split(
        rows,
        test_size=0.10,
        random_state=SEED,
        shuffle=True,
    )

    train, val = train_test_split(
        train_val,
        test_size=0.111,
        random_state=SEED,
        shuffle=True,
    )

    log(f"Toplam veri: {len(rows)}")
    log(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    # ------------------------------------------------------------
    # Model seçimi
    # ------------------------------------------------------------
    if (
        RESUME_FROM_BEST_MODEL_IF_EXISTS
        and os.path.isdir(BEST_MODEL_DIR)
        and os.path.exists(os.path.join(BEST_MODEL_DIR, "config.json"))
    ):
        model_source = BEST_MODEL_DIR
        log(f"Mevcut kayıtlı modelden devam ediliyor: {BEST_MODEL_DIR}")
    else:
        model_source = BASE_MODEL_NAME
        log(f"Base model yükleniyor: {BASE_MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_source)
    model.to(device)

    if hasattr(model, "generation_config"):
        model.generation_config.forced_bos_token_id = None

    if USE_GRADIENT_CHECKPOINTING:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        log("Gradient checkpointing aktif.")

    log("Model hazır.")
    log(f"GPU durumu: {gpu_info()}")

    # ------------------------------------------------------------
    # Dataset / DataLoader
    # ------------------------------------------------------------
    train_dataset = HikayeDataset(train, tokenizer, MAX_INPUT, MAX_TARGET)
    val_dataset = HikayeDataset(val, tokenizer, MAX_INPUT, MAX_TARGET)

    use_fp16 = torch.cuda.is_available()

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if use_fp16 else None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=data_collator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=data_collator,
    )

    log(f"Train batch/epoch: {len(train_loader)}")
    log(f"Gradient accumulation: {GRAD_ACCUM}")
    log(f"Efektif batch size yaklaşık: {BATCH_SIZE * GRAD_ACCUM}")

    # ------------------------------------------------------------
    # Optimizer / Scheduler
    # ------------------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    update_steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM)
    total_training_steps = update_steps_per_epoch * EPOCHS
    actual_warmup_steps = min(WARMUP_STEPS, max(0, total_training_steps // 10))

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=actual_warmup_steps,
        num_training_steps=total_training_steps,
    )

    log(f"Total optimizer update steps: {total_training_steps}")
    log(f"Warmup steps: {actual_warmup_steps}")

    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    # ------------------------------------------------------------
    # Eğitim
    # ------------------------------------------------------------
    best_val_loss = float("inf")
    no_improve_count = 0

    log("\nEğitim başlıyor...")
    log("=" * 70)

    global_update_step = 0

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        model.train()
        train_loss_sum = 0.0
        train_batch_count = 0

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            batch = {
                k: v.to(device, non_blocking=True)
                for k, v in batch.items()
            }

            with torch.cuda.amp.autocast(enabled=use_fp16):
                outputs = model(**batch)
                raw_loss = outputs.loss
                loss = raw_loss / GRAD_ACCUM

            scaler.scale(loss).backward()

            train_loss_sum += raw_loss.item()
            train_batch_count += 1

            do_update = (step % GRAD_ACCUM == 0) or (step == len(train_loader))

            if do_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_update_step += 1

            if step % LOG_EVERY == 0:
                avg_loss = train_loss_sum / train_batch_count
                current_lr = scheduler.get_last_lr()[0]
                elapsed_min = (time.time() - epoch_start) / 60

                log(
                    f"Epoch {epoch}/{EPOCHS} | "
                    f"Adım {step}/{len(train_loader)} | "
                    f"Train Loss: {avg_loss:.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Süre: {elapsed_min:.1f} dk | "
                    f"{gpu_info()}"
                )

        avg_train_loss = train_loss_sum / max(1, train_batch_count)

        # --------------------------------------------------------
        # Validasyon
        # --------------------------------------------------------
        val_start = time.time()
        val_loss = evaluate(model, val_loader, device, use_fp16)
        val_min = (time.time() - val_start) / 60

        epoch_min = (time.time() - epoch_start) / 60

        log("\n" + "-" * 70)
        log(
            f"Epoch {epoch}/{EPOCHS} tamamlandı | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Epoch Süresi: {epoch_min:.1f} dk | "
            f"Val Süresi: {val_min:.1f} dk"
        )

        # --------------------------------------------------------
        # En iyi modeli kaydet
        # --------------------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0

            os.makedirs(BEST_MODEL_DIR, exist_ok=True)

            model.save_pretrained(BEST_MODEL_DIR, safe_serialization=True)
            tokenizer.save_pretrained(BEST_MODEL_DIR)

            log(f"--> En iyi model kaydedildi: {BEST_MODEL_DIR}")
            log(f"--> Yeni best val loss: {best_val_loss:.4f}")

        else:
            no_improve_count += 1
            log(f"Val loss iyileşmedi. Sayaç: {no_improve_count}/{EARLY_STOPPING_PATIENCE}")

        # --------------------------------------------------------
        # İsteğe bağlı training state
        # --------------------------------------------------------
        if SAVE_TRAINING_STATE:
            os.makedirs(LAST_STATE_DIR, exist_ok=True)

            torch.save(
                {
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if use_fp16 else None,
                },
                os.path.join(LAST_STATE_DIR, "trainer_state.pt"),
            )

            log(f"Training state kaydedildi: {LAST_STATE_DIR}")

        clean_memory()
        log(f"Epoch sonu GPU durumu: {gpu_info()}")
        log("-" * 70 + "\n")

        # --------------------------------------------------------
        # Early stopping
        # --------------------------------------------------------
        if no_improve_count >= EARLY_STOPPING_PATIENCE:
            log("Early stopping tetiklendi. Eğitim durduruluyor.")
            break

    log("=" * 70)
    log(f"Eğitim bitti. En iyi val loss: {best_val_loss:.4f}")
    log(f"En iyi model klasörü: {BEST_MODEL_DIR}")
    log("=" * 70)

    # ------------------------------------------------------------
    # En iyi modeli yükle ve örnek tahmin yap
    # ------------------------------------------------------------
    clean_memory()

    log("\nEn iyi model yükleniyor ve örnek tahmin yapılıyor...")

    best_model = AutoModelForSeq2SeqLM.from_pretrained(BEST_MODEL_DIR)
    best_model.to(device)

    if USE_GRADIENT_CHECKPOINTING:
        best_model.config.use_cache = False

    show_sample_prediction(best_model, tokenizer, test[0], device)

    clean_memory()


# ============================================================
# 8) ÇALIŞTIR
# ============================================================

if __name__ == "__main__":
    main()