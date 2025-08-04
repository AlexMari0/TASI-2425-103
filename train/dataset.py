import datasets
import random
import pandas as pd
import torch
from transformers import AutoTokenizer
from typing import Mapping, Tuple

# --- PERBAIKAN DI SINI ---
# Pindahkan import spacy ke dalam kelas QAEvalDataset
# atau pastikan Anda memiliki model bahasa yang benar terinstal dan diimpor.
# en_core_web_sm adalah model bahasa Inggris kecil.
# xx_ent_wiki_sm adalah model multi-bahasa yang lebih besar.
# Pilih yang sesuai dengan kebutuhan Anda.
# Jika QAEvalDataset tidak digunakan untuk pelatihan QG, bagian ini bisa diabaikan/dihapus.
# Jika digunakan, pastikan 'pip install spacy' dan 'python -m spacy download en_core_web_sm' (atau xx_ent_wiki_sm) telah dijalankan.
try:
    import spacy
    # Hapus import en_core_web_sm karena spacy.load yang sebenarnya digunakan
    # adalah 'xx_ent_wiki_sm' atau 'en_core_web_sm'
except ImportError:
    print("Warning: spacy not installed. QAEvalDataset might not function correctly if used.")
    spacy = None
# --- AKHIR PERBAIKAN ---


class QGDataset(torch.utils.data.Dataset):
    def __init__(self, data: datasets.Dataset, max_length: int, pad_mask_id: int, tokenizer: AutoTokenizer) -> None:
        self.data = pd.DataFrame({
            "context": [item["context"] for item in data],
            "question": [item["question"] for item in data],
            # --- PERBAIKAN DI SINI ---
            "answer": [item["answer"] for item in data] # Memastikan kolom 'answer' dimuat dari datasets.Dataset
            # --- AKHIR PERBAIKAN ---
        })
        self.max_length = max_length
        self.pad_mask_id = pad_mask_id
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        item = self.data.loc[index]
        
        # --- PERBAIKAN DI SINI ---
        # Menggabungkan konteks dan jawaban untuk input encoder
        # Format: "<answer> {jawaban_teks} <context> {konteks_teks}"
        # Token khusus <answer> dan <context> harus sudah ditambahkan ke tokenizer di main.py
        
        # Pastikan item.answer bukan None atau string kosong, konversi ke string
        answer_text = str(item.answer) if item.answer else "" 

        # Ini adalah input yang akan digunakan oleh ENCODER
        encoder_input_text = f"<answer> {answer_text} <context> {item.context}"
        
        input_ids, attention_mask = self._encode_text(encoder_input_text)
        # --- AKHIR PERBAIKAN ---
        
        # Labels (output target) tetap hanya pertanyaan
        labels, _ = self._encode_text(item.question)
        masked_labels = self._mask_label_padding(labels)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": masked_labels
        }

    def _encode_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_text = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        # squeeze() untuk menghilangkan dimensi batch tambahan (misal: dari [1, seq_len] menjadi [seq_len])
        return encoded_text["input_ids"].squeeze(), encoded_text["attention_mask"].squeeze()

    def _mask_label_padding(self, labels: torch.Tensor) -> torch.Tensor:
        # Mengganti token ID padding dengan pad_mask_id (biasanya -100)
        # agar tidak dihitung dalam perhitungan loss oleh model.
        labels[labels == self.tokenizer.pad_token_id] = self.pad_mask_id
        return labels

class QAEvalDataset(torch.utils.data.Dataset):
    def __init__(self, data: datasets.Dataset, max_length: int, tokenizer: AutoTokenizer) -> None:
        # Konversi dataset SQuAD v2 ke DataFrame
        self.data = pd.DataFrame({
            "question": [item["question"] for item in data],
            "answer": [item["answers"]["text"][0] if len(item["answers"]["text"]) > 0 else "" for item in data]
        })
        self.max_length = max_length
        self.transforms = [self.shuffle, self.corrupt]
        self.hf_tokenizer = tokenizer
        
        # --- PERBAIKAN DI SINI ---
        # Memuat model spacy di sini agar tidak diimpor global
        # Gunakan model yang sesuai. en_core_web_sm adalah pilihan umum.
        # Pastikan Anda telah mengunduh model: python -m spacy download en_core_web_sm
        if spacy: # Cek apakah spacy berhasil diimpor
            try:
                self.spacy_tokenizer = spacy.load("en_core_web_sm") # Mengubah ke en_core_web_sm atau yang sesuai
            except Exception as e:
                print(f"Error loading spacy model: {e}. QAEvalDataset will not function correctly for corruption.")
                self.spacy_tokenizer = None
        else:
            self.spacy_tokenizer = None
        # --- AKHIR PERBAIKAN ---

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        item = self.data.loc[index] # Menggunakan .loc[index] untuk akses baris DataFrame
        question = str(item["question"]) # Pastikan ini string
        answer = str(item["answer"]) # Pastikan ini string

        label = 1 if answer else 0  # Jika tidak ada jawaban, label = 0
        if label == 0:
            # Pastikan transforms (shuffle/corrupt) hanya dipanggil jika spacy_tokenizer ada
            if self.spacy_tokenizer and self.transforms:
                question, answer = random.choice(self.transforms)(question, answer)
            else:
                # Fallback jika spacy tidak tersedia
                question, answer = self.shuffle(question, answer) # Gunakan shuffle sebagai fallback

        encoded_data = self.hf_tokenizer(
            text=question,
            text_pair=answer,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded_data["input_ids"].squeeze(),
            "attention_mask": encoded_data["attention_mask"].squeeze(),
            "token_type_ids": encoded_data["token_type_ids"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.int64)
        }

    def shuffle(self, question: str, answer: str) -> Tuple[str, str]:
        shuffled_answer = answer
        # Pastikan tidak mengulang jawaban yang sama jika hanya ada satu
        if len(self.data['answer'].unique()) > 1:
            while shuffled_answer == answer:
                shuffled_answer = self.data.sample(1)['answer'].item()
        return question, shuffled_answer

    def corrupt(self, question: str, answer: str) -> Tuple[str, str]:
        # --- PERBAIKAN DI SINI ---
        if not self.spacy_tokenizer: # Periksa apakah spacy tokenizer berhasil dimuat
            return self.shuffle(question, answer) # Fallback jika spacy tidak ada

        doc = self.spacy_tokenizer(question)
        if len(doc.ents) > 1:
            copy_ent = str(random.choice(doc.ents))
            for ent in doc.ents:
                question = question.replace(str(ent), copy_ent)
        elif len(doc.ents) == 1:
            # Jika hanya ada satu entitas, ganti jawaban dengan entitas tersebut
            # Pastikan ini adalah perilaku yang Anda inginkan untuk "corrupt"
            answer = str(doc.ents[0])
        else:
            # Jika tidak ada entitas, lakukan shuffle sebagai fallback
            question, answer = self.shuffle(question, answer)
        return question, answer
        # --- AKHIR PERBAIKAN ---