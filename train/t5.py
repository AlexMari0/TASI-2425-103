import argparse
import datasets
import torch
import json
import gc
import os
import pandas as pd
# from sklearn.model_selection import train_test_split # Tidak digunakan untuk splitting di main_t5.py

# Pastikan Anda memiliki versi terbaru dari dataset.py dan trainer.py yang sudah kita modifikasi
from dataset import QGDataset 
from trainer import Trainer 
from transformers import T5ForConditionalGeneration, AutoTokenizer, EncoderDecoderConfig # EncoderDecoderConfig tidak diperlukan untuk T5

torch.multiprocessing.set_sharing_strategy('file_system')

# ============================ #
#        Argument Parsing      #
# ============================ #
# Penjelasan: Fungsi ini mendefinisikan dan menguraikan argumen baris perintah
# yang dapat digunakan untuk mengkonfigurasi proses pelatihan model T5.
# Ini mencakup pengaturan hyperparameter seperti jumlah epoch, learning rate, ukuran batch,
# dan jalur ke model pra-terlatih serta direktori penyimpanan.
# Perangkat (CPU/GPU) juga ditentukan secara otomatis atau dapat dispesifikasikan.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10) # Mengubah epoch menjadi 10
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--valid_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1) # Disesuaikan ke 1 jika tidak ada akumulasi spesifik di Trainer

    # Menggunakan model T5. Pastikan ini adalah path yang benar ke model T5 Anda,
    # atau model ID dari Hugging Face (misal: "t5-small", "t5-base").
    # Jika Anda melanjutkan training dari '/home/samuel/tasi2425103/fine-tuning/hasil-pisa/t5/',
    # pastikan model ini sudah ada di sana.
    parser.add_argument("--qg_model", type=str, default="allenai/t5-small-squad2-question-generation") # Contoh: gunakan t5-small dari HF
    # parser.add_argument("--qg_model", type=str, default="/home/samuel/tasi2425103/fine-tuning/hasil-pisa/t5/") # Jika melanjutkan dari checkpoint lokal

    parser.add_argument("--pad_mask_id", type=int, default=-100)
    parser.add_argument("--save_dir", type=str, default="/home/samuel/tasi2425103/revisi/model/final_t5/") # Ubah save_dir untuk T5
    
    # Jalur ke dataset yang sudah di-split
    parser.add_argument("--train_pisa_path", type=str, default="/home/samuel/tasi2425103/revisi/split/pisa/train_pisa.json")
    parser.add_argument("--valid_pisa_path", type=str, default="/home/samuel/tasi2425103/revisi/split/pisa/valid_pisa.json")
    parser.add_argument("--train_squad_path", type=str, default="/home/samuel/tasi2425103/revisi/split/squad/train_squad.json")
    parser.add_argument("--valid_squad_path", type=str, default="/home/samuel/tasi2425103/revisi/split/squad/valid_squad.json")

    device = "cuda" if torch.cuda.is_available() else "cpu" # Default ke "cuda", biarkan PyTorch yang memilih indeks
    parser.add_argument("--device", type=str, default=device)

    parser.add_argument("--dataloader_workers", type=int, default=0)
    parser.add_argument("--early_stopping_patience", type=int, default=3) # Menambahkan argumen early stopping
    return parser.parse_args()

# ============================ #
#      Tokenizer Initialization      #
# ============================ #
# Penjelasan: Fungsi ini bertanggung jawab untuk memuat tokenizer pra-terlatih
# dari checkpoint model T5 yang diberikan. Ini juga menambahkan token khusus
# (<answer>, <context>) ke tokenizer, yang penting untuk format input model Question Generation.
def get_tokenizer(checkpoint: str):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<answer>', '<context>']})
    return tokenizer

# ============================ #
#       Model Initialization       #
# ============================ #
# Penjelasan: Fungsi ini memuat model T5ForConditionalGeneration pra-terlatih
# dari checkpoint yang ditentukan. Ini menyesuaikan ukuran embedding token
# agar sesuai dengan tokenizer yang diperluas (karena penambahan token khusus)
# dan mengaktifkan gradient checkpointing untuk penghematan memori.
# Model kemudian dipindahkan ke perangkat yang ditentukan (CPU/GPU).
def get_model(checkpoint: str, device: str, tokenizer) -> T5ForConditionalGeneration:
    model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    model.resize_token_embeddings(len(tokenizer)) # T5 tidak memiliki encoder/decoder terpisah untuk resize
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model.to(device)

# ============================ #
#        Load & Combine Datasets      #
# ============================ #
# Penjelasan: Fungsi ini memuat dataset training dan validasi dari file JSON
# yang sudah di-split (PISA dan SQuAD) secara terpisah, kemudian menggabungkannya.
# Ini memastikan data disiapkan dengan kolom 'context', 'question', dan 'answer'
# agar dapat digunakan oleh QGDataset untuk pelatihan Question Generation.
def load_and_combine_datasets(
    train_pisa_path: str,
    valid_pisa_path: str,
    train_squad_path: str,
    valid_squad_path: str
) -> datasets.DatasetDict:

    # Fungsi bantu untuk memuat dan memproses satu file JSON split
    def _load_single_split(file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        data_list = []
        for item in raw_data:
            question_list = item.get("question", [""])  
            answer_list = item.get("answer", [""])

            context_text = item.get("context", "")
            question_text = " || ".join([str(q) for q in question_list])
            answer_text = " || ".join([str(a) for a in answer_list]) 
            
            data_list.append({
                "context": context_text,
                "question": question_text,
                "answer": answer_text 
            })
        return pd.DataFrame(data_list)

    print(f"Memuat dataset training PISA dari: {train_pisa_path}")
    train_df_pisa = _load_single_split(train_pisa_path)
    print(f"Memuat dataset validasi PISA dari: {valid_pisa_path}")
    valid_df_pisa = _load_single_split(valid_pisa_path)

    print(f"Memuat dataset training SQuAD dari: {train_squad_path}")
    train_df_squad = _load_single_split(train_squad_path)
    print(f"Memuat dataset validasi SQuAD dari: {valid_squad_path}")
    valid_df_squad = _load_single_split(valid_squad_path)

    # Gabungkan DataFrame training dan validasi
    combined_train_df = pd.concat([train_df_pisa, train_df_squad], ignore_index=True)
    combined_valid_df = pd.concat([valid_df_pisa, valid_df_squad], ignore_index=True)

    print(f"Total data training gabungan: {len(combined_train_df)}")
    print(f"Total data validasi gabungan: {len(combined_valid_df)}")

    return datasets.DatasetDict({
        "train": datasets.Dataset.from_pandas(combined_train_df),
        "validation": datasets.Dataset.from_pandas(combined_valid_df)
    })

# ============================ #
#           Main Execution           #
# ============================ #
# Penjelasan: Ini adalah titik masuk utama dari skrip. Ini menguraikan argumen,
# menginisialisasi tokenizer dan model T5, memuat dan menggabungkan dataset
# training/validasi, membuat instance `QGDataset` dan `Trainer`, lalu memulai
# proses pelatihan. Setelah pelatihan, ini melakukan pembersihan memori GPU
# dan mencetak pesan konfirmasi penyimpanan model dan riwayat loss.
if __name__ == "__main__":
    args = parse_args()
    print(f"Using device: {args.device}")

    # Penting: T5 menggunakan tokenizer yang sama untuk encoder dan decoder.
    # qg_model di sini akan menjadi checkpoint awal T5, bukan bert2bert.
    tokenizer = get_tokenizer(args.qg_model) 
    
    # Memuat dan menggabungkan dataset training dan validasi
    dataset = load_and_combine_datasets(
        train_pisa_path=args.train_pisa_path,
        valid_pisa_path=args.valid_pisa_path,
        train_squad_path=args.train_squad_path,
        valid_squad_path=args.valid_squad_path
    )

    # train_set dan valid_set dibuat dengan dataset yang memiliki kolom 'answer'
    train_set = QGDataset(dataset["train"], args.max_length, args.pad_mask_id, tokenizer)
    valid_set = QGDataset(dataset["validation"], args.max_length, args.pad_mask_id, tokenizer)

    # Memuat model T5
    model = get_model(args.qg_model, args.device, tokenizer)

    trainer = Trainer(
        dataloader_workers=args.dataloader_workers,
        device=args.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model=model,
        save_dir=args.save_dir,
        tokenizer=tokenizer,
        train_batch_size=args.train_batch_size,
        train_set=train_set,
        valid_batch_size=args.valid_batch_size,
        valid_set=valid_set,
        early_stopping_patience=args.early_stopping_patience # Meneruskan patience ke Trainer
    )

    trainer.train()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"Proses training selesai. Model terbaik (jika early stopping aktif) dan riwayat loss disimpan di {args.save_dir}")