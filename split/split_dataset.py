import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import datasets

def split_and_save_dataset(
    dataset_input_path: str,
    split_output_dir: str,
    dataset_type: str,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_seed: int = 42
):
    """
    Memuat dataset, mengelompokkan entri berdasarkan konteks unik (jika tipe 'pisa'),
    membagi data berdasarkan TITLE unik, lalu menyimpan hasilnya,
    dengan hasil split diurutkan berdasarkan title dan nama file disesuaikan.

    Args:
        dataset_input_path (str): Jalur lengkap ke file JSON dataset asli Anda.
        split_output_dir (str): Direktori untuk menyimpan file JSON hasil split.
        dataset_type (str): Tipe dataset, saat ini 'pisa' atau 'squad' yang didukung.
        test_size (float): Proporsi dataset yang akan digunakan untuk set test.
        val_size (float): Proporsi dataset yang akan digunakan untuk set validation (dari total asli).
        random_seed (int): Seed untuk reproduktifitas pembagian data.
    """

    # --- 1. Pemuatan dan Pra-pemrosesan Data Awal ---
    print(f"\n--- Memulai proses untuk dataset: {os.path.basename(dataset_input_path)} ---")
    print(f"Memuat dataset dari: {dataset_input_path}")
    with open(dataset_input_path, "r", encoding="utf-8") as f:
        raw_data_from_file = json.load(f)
        
        # Penyesuaian struktur akar JSON berdasarkan tipe dataset
        if dataset_type == "pisa":
            if isinstance(raw_data_from_file, dict) and raw_data_from_file:
                raw_data_list = list(raw_data_from_file.values())[0]
            elif isinstance(raw_data_from_file, list):
                raw_data_list = raw_data_from_file
            else:
                raise ValueError(f"Struktur JSON PISA tidak dikenali untuk {dataset_input_path}.")
        elif dataset_type == "squad":
            if isinstance(raw_data_from_file, dict):
                raw_data_list = list(raw_data_from_file.values())
            elif isinstance(raw_data_from_file, list):
                raw_data_list = raw_data_from_file
            else:
                raise ValueError(f"Struktur JSON SQUAD tidak dikenali untuk {dataset_input_path}.")
        else:
            raise ValueError("dataset_type tidak valid. Harus 'pisa' atau 'squad'.")


    # Fungsi bantu untuk pra-pemrosesan teks
    def preprocess_text(text):
        if not isinstance(text, str):
            return ""
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    # Struktur untuk mengisi DataFrame master
    data_for_df = [] 

    if dataset_type == "pisa":
        grouped_data_by_context = {}
        for item in raw_data_list:
            context = preprocess_text(str(item.get("context", "")))
            title = preprocess_text(str(item.get("title", "")))
            question = preprocess_text(str(item.get("question", "")))
            answer = preprocess_text(str(item.get("answers", "")))
            
            # Filter entri jika context, question, atau answer kosong setelah pre-process
            if not context or not question or not answer:
                continue

            if context not in grouped_data_by_context:
                grouped_data_by_context[context] = {
                    "title": title,
                    "context": context,
                    "questions": [],
                    "answers": []
                }
            
            grouped_data_by_context[context]["questions"].append(question)
            grouped_data_by_context[context]["answers"].append(answer)
        
        for context_key, data in grouped_data_by_context.items():
            data_for_df.append({
                "title": data["title"],
                "context": data["context"],
                "raw_questions": data["questions"],
                "raw_answers": data["answers"]
            })
        df_master = pd.DataFrame(data_for_df)

    elif dataset_type == "squad":
        for item in raw_data_list:
            answer_text = item.get("answer", "") # Kunci 'answer' untuk SQuAD
            
            context = preprocess_text(item.get("context", ""))
            title = preprocess_text(item.get("title", ""))
            question = preprocess_text(item.get("question", ""))
            answer = preprocess_text(answer_text)

            # Filter entri jika context, question, atau answer kosong setelah pre-process
            if not context or not question or not answer:
                continue
            
            data_for_df.append({
                "title": title,
                "context": context,
                "raw_questions": [question],
                "raw_answers": [answer]
            })
        df_master = pd.DataFrame(data_for_df)
    
    print(f"Total entri yang diproses untuk pembagian: {len(df_master)}")

    # --- 2. Pembagian Dataset Berdasarkan Title ---
    unique_titles = df_master['title'].unique()
    print(f"Total judul unik yang akan dibagi: {len(unique_titles)}")

    if len(unique_titles) < 3:
        print("PERINGATAN: Jumlah judul unik terlalu sedikit untuk pembagian 80/10/10 yang stabil.")
        print("Semua data mungkin akan masuk ke set train atau proporsi split akan sangat tidak seimbang.")

    train_val_titles, test_titles = train_test_split(unique_titles, test_size=test_size, random_state=random_seed)
    train_titles, valid_titles = train_test_split(train_val_titles, test_size=val_size / (1 - test_size), random_state=random_seed)

    print(f"Jumlah judul Train: {len(train_titles)}")
    print(f"Jumlah judul Validation: {len(valid_titles)}")
    print(f"Jumlah judul Test: {len(test_titles)}")

    train_df = df_master[df_master['title'].isin(train_titles)].copy()
    valid_df = df_master[df_master['title'].isin(valid_titles)].copy()
    test_df = df_master[df_master['title'].isin(test_titles)].copy()

    print(f"\nJumlah data (entri unik konteks/QA) dalam split:")
    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(valid_df)}")
    print(f"Test: {len(test_df)}")

    # --- 3. Penyimpanan Data Hasil Split ---
    os.makedirs(split_output_dir, exist_ok=True)

    def save_df_to_json(df_to_save, split_name: str, dataset_prefix: str):
        # Nama file baru: train_pisa.json, valid_squad.json, dll.
        filename = f"{split_name}_{dataset_prefix}.json"

        df_sorted = df_to_save.sort_values(by='title', ascending=True).reset_index(drop=True)

        data_to_dump = []
        for _, row in df_sorted.iterrows():
            data_to_dump.append({
                "title": row["title"],
                "context": row["context"],
                "question": row["raw_questions"],
                "answer": row["raw_answers"]
            })
        file_path = os.path.join(split_output_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data_to_dump, f, ensure_ascii=False, indent=2)
        print(f"Split '{filename}' disimpan di: {file_path}")

    # Panggil fungsi save_df_to_json dengan nama file yang disesuaikan
    save_df_to_json(train_df, "train", dataset_type)
    save_df_to_json(valid_df, "valid", dataset_type)
    save_df_to_json(test_df, "test", dataset_type)

    # --- Opsional: Mengembalikan DatasetDict untuk alur kerja berikutnya ---
    train_df_flat = train_df.copy()
    valid_df_flat = valid_df.copy()

    train_df_flat["question"] = train_df_flat["raw_questions"].apply(lambda x: " || ".join(x))
    train_df_flat["answer"] = train_df_flat["raw_answers"].apply(lambda x: " || ".join(x))
    valid_df_flat["question"] = valid_df_flat["raw_questions"].apply(lambda x: " || ".join(x))
    valid_df_flat["answer"] = valid_df_flat["raw_answers"].apply(lambda x: " || ".join(x))

    train_df_flat = train_df_flat.drop(columns=["raw_questions", "raw_answers"])
    valid_df_flat = valid_df_flat.drop(columns=["raw_questions", "raw_answers"])

    for df_temp in [train_df_flat, valid_df_flat]:
        df_temp["title"] = df_temp["title"].astype(str)
        df_temp["context"] = df_temp["context"].astype(str)
        df_temp["question"] = df_temp["question"].astype(str)
        df_temp["answer"] = df_temp["answer"].astype(str)

    # Mengembalikan DatasetDict hanya untuk train dan validation
    return datasets.DatasetDict({
        "train": datasets.Dataset.from_pandas(train_df_flat),
        "validation": datasets.Dataset.from_pandas(valid_df_flat)
    })

# --- Fungsi Main untuk Menjalankan Split Kedua Dataset ---
def main():
    base_dir = "/home/samuel/tasi2425103/revisi"

    # Konfigurasi untuk Indopisa
    print("--- Memproses Dataset PISA ---")
    split_and_save_dataset(
        dataset_input_path=os.path.join(base_dir, "dataset", "indo_pisa.json"),
        split_output_dir=os.path.join(base_dir, "split", "pisa"),
        dataset_type="pisa"
    )
    print("Proses split Dataset PISA selesai!")

    # Konfigurasi untuk IndoSquad
    print("\n--- Memproses Dataset SQUAD ---")
    split_and_save_dataset(
        dataset_input_path=os.path.join(base_dir, "dataset", "indo_squad.json"),
        split_output_dir=os.path.join(base_dir, "split", "squad"),
        dataset_type="squad"
    )
    print("Proses split Dataset SQUAD selesai!")

# --- Eksekusi ---
if __name__ == "__main__":
    main()