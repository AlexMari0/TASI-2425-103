import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import os # Import os untuk membuat direktori
import json # Import json untuk menyimpan riwayat loss

from utils import AverageMeter # Pastikan utils.py dengan AverageMeter ada

class Trainer:
    # ============================ #
    #       Initialization       #
    # ============================ #
    # Penjelasan: Konstruktor ini menginisialisasi semua parameter dan komponen
    # yang diperlukan untuk proses pelatihan. Ini termasuk perangkat (device),
    # hyperparameter pelatihan (epochs, learning_rate, batch_size), model,
    # tokenizer, direktori penyimpanan, serta DataLoader untuk data train dan validasi.
    # Selain itu, di sini diatur optimizer, scheduler learning rate, dan parameter
    # untuk early stopping serta pencatatan riwayat loss.
    def __init__(
        self,
        dataloader_workers: int,
        device: str,
        epochs: int,
        learning_rate: float,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        save_dir: str,
        train_batch_size: int,
        train_set: Dataset,
        valid_batch_size: int,
        valid_set: Dataset,
        evaluate_on_accuracy: bool = False,
        early_stopping_patience: int = None, # Tambahkan argumen patience untuk early stopping
    ) -> None:
        self.device = device
        self.epochs = epochs
        self.save_dir = save_dir
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.train_loader = DataLoader(
            train_set,
            batch_size=train_batch_size,
            num_workers=dataloader_workers,
            shuffle=True
        )
        self.valid_loader = DataLoader(
            valid_set,
            batch_size=valid_batch_size, # Pastikan ini menggunakan valid_batch_size
            num_workers=dataloader_workers,
            shuffle=False
        )
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        num_training_steps = len(self.train_loader) * epochs
        num_warmup_steps = int(0.1 * num_training_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        self.train_loss_meter = AverageMeter() # Ganti nama agar lebih jelas
        self.evaluate_on_accuracy = evaluate_on_accuracy
        
        # Inisialisasi untuk early stopping dan pencatatan riwayat
        # best_valid_score diinisialisasi berdasarkan apakah kita mengevaluasi akurasi atau loss
        self.best_valid_score = float("inf") if not evaluate_on_accuracy else 0 
        self.early_stopping_patience = early_stopping_patience
        self.patience_counter = 0
        self.history = {'train_loss': [], 'valid_loss': []} # Untuk menyimpan riwayat loss

    # ============================ #
    #         Training Loop        #
    # ============================ #
    # Penjelasan: Metode ini menjalankan seluruh proses pelatihan model.
    # Ini melakukan iterasi melalui setiap epoch, menjalankan fase pelatihan
    # dan validasi. Selama pelatihan, model diatur ke mode train, gradien di-reset,
    # data dipindahkan ke perangkat yang benar, dan forward/backward pass dilakukan.
    # Loss dihitung dan diakumulasikan. Setelah setiap epoch, model dievaluasi
    # pada set validasi, dan metrik (loss atau akurasi) dicatat.
    # Logika early stopping diimplementasikan di sini untuk menghentikan pelatihan
    # lebih awal jika metrik validasi tidak membaik setelah sejumlah epoch tertentu.
    # Riwayat loss disimpan setelah pelatihan selesai.
    def train(self) -> None:
        print("Memulai proses training...")
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self.train_loss_meter.reset() # Reset average meter untuk setiap epoch

            with tqdm(total=len(self.train_loader), unit="batches", desc=f"Epoch {epoch}/{self.epochs} [Train]") as tepoch:
                for data in self.train_loader:
                    self.optimizer.zero_grad()
                    data = {key: value.to(self.device) for key, value in data.items()}
                    output = self.model(**data)
                    loss = output.loss
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.train_loss_meter.update(loss.item(), self.train_batch_size)
                    tepoch.set_postfix({"train_loss": self.train_loss_meter.avg, "lr": self.scheduler.get_last_lr()[0]})
                    tepoch.update(1)
            
            current_train_loss = self.train_loss_meter.avg
            
            # Evaluasi pada validasi
            if self.evaluate_on_accuracy:
                current_valid_score = self.evaluate_accuracy(self.valid_loader)
                metric_name = "Accuracy"
            else:
                current_valid_score = self.evaluate_loss(self.valid_loader) # Menggunakan evaluate_loss
                metric_name = "Loss"

            self.history['train_loss'].append(current_train_loss)
            self.history['valid_loss'].append(current_valid_score) # valid_score adalah loss jika evaluate_on_accuracy=False

            print(f"Epoch {epoch}/{self.epochs} - Train Loss: {current_train_loss:.4f} - Valid {metric_name}: {current_valid_score:.4f}")

            # Logika Early Stopping
            if self.early_stopping_patience is not None: # Pastikan early stopping aktif
                if self.evaluate_on_accuracy: # Untuk akurasi, kita ingin score yang lebih tinggi
                    if current_valid_score > self.best_valid_score:
                        print(f"Validation {metric_name} improved from {self.best_valid_score:.4f} to {current_valid_score:.4f}. Saving model.")
                        self.best_valid_score = current_valid_score
                        self.patience_counter = 0
                        self._save_model() # Menyimpan model terbaik
                    else:
                        self.patience_counter += 1
                        print(f"Validation {metric_name} did not improve. Patience: {self.patience_counter}/{self.early_stopping_patience}")
                        if self.patience_counter >= self.early_stopping_patience:
                            print(f"Early stopping triggered after {epoch} epochs.")
                            break # Menghentikan training
                else: # Untuk loss, kita ingin loss yang lebih rendah
                    if current_valid_score < self.best_valid_score:
                        print(f"Validation {metric_name} decreased from {self.best_valid_score:.4f} to {current_valid_score:.4f}. Saving model.")
                        self.best_valid_score = current_valid_score
                        self.patience_counter = 0
                        self._save_model() # Menyimpan model terbaik
                    else:
                        self.patience_counter += 1
                        print(f"Validation {metric_name} did not decrease. Patience: {self.patience_counter}/{self.early_stopping_patience}")
                        if self.patience_counter >= self.early_stopping_patience:
                            print(f"Early stopping triggered after {epoch} epochs.")
                            break # Menghentikan training
        
        print("Proses training selesai.")
        self._save_loss_history() # Simpan riwayat loss setelah training selesai

    # ============================ #
    #       Loss Evaluation        #
    # ============================ #
    # Penjelasan: Metode ini mengevaluasi performa model pada set data yang diberikan
    # berdasarkan loss. Model diatur ke mode evaluasi, dan perhitungan gradien dinonaktifkan.
    # Ini menghitung rata-rata loss dari semua batch di DataLoader.
    @torch.no_grad()
    def evaluate_loss(self, dataloader: DataLoader) -> float: # Ganti nama fungsi dari evaluate
        self.model.eval()
        eval_loss_meter = AverageMeter()
        with tqdm(total=len(dataloader), unit="batches", desc="Validation Loss") as tepoch:
            for data in dataloader:
                data = {key: value.to(self.device) for key, value in data.items()}
                output = self.model(**data)
                loss = output.loss
                eval_loss_meter.update(loss.item(), self.valid_batch_size)
                tepoch.set_postfix({"valid_loss": eval_loss_meter.avg})
                tepoch.update(1)
        return eval_loss_meter.avg

    # ============================ #
    #      Accuracy Evaluation     #
    # ============================ #
    # Penjelasan: Metode ini mengevaluasi performa model pada set data yang diberikan
    # berdasarkan akurasi. Ini umumnya digunakan untuk tugas klasifikasi.
    # **Penting**: Untuk tugas Question Generation (seq2seq), akurasi token-per-token
    # seperti ini mungkin bukan metrik evaluasi yang paling informatif atau tepat.
    # Metrik seperti BLEU, ROUGE, atau METEOR lebih umum digunakan untuk mengevaluasi
    # kualitas teks yang dihasilkan. Jika model Anda adalah QG, Anda mungkin ingin
    # mempertimbangkan untuk menonaktifkan `evaluate_on_accuracy` atau menggantinya
    # dengan metrik generation.
    @torch.no_grad()
    def evaluate_accuracy(self, dataloader: DataLoader) -> float:
        self.model.eval()
        accuracy_meter = AverageMeter()
        with tqdm(total=len(dataloader), unit="batches", desc="Validation Accuracy") as tepoch:
            for data in dataloader:
                data = {key: value.to(self.device) for key, value in data.items()}
                output = self.model(**data)
                
                # Untuk model EncoderDecoder, output.logits biasanya berbentuk (batch_size, sequence_length, vocab_size)
                # Jika Anda mengukur akurasi token per token, Anda perlu membandingkan logits dengan labels.
                # Namun, untuk tugas Question Generation, evaluasi loss biasanya lebih umum.
                # Jika ini adalah model klasifikasi lain, logika ini mungkin tepat.
                # Asumsi di sini: output.logits adalah untuk klasifikasi (misal: binary classification)
                preds = torch.argmax(output.logits, dim=1) 
                
                # Perhatian: Anda perlu memastikan `data["labels"]` sesuai dengan `preds` ini.
                # Untuk QG, `labels` adalah token ID dari pertanyaan target.
                # Mengukur akurasi seperti ini untuk QG biasanya tidak langsung.
                # Anda mungkin perlu menyesuaikan ini atau menghapus `evaluate_on_accuracy` jika tidak relevan.
                score = accuracy_score(data["labels"].cpu().view(-1), preds.cpu().view(-1)) 
                accuracy_meter.update(score, self.valid_batch_size)
                tepoch.set_postfix({"valid_acc": accuracy_meter.avg})
                tepoch.update(1)
        return accuracy_meter.avg

    # ============================ #
    #        Save Model Logic      #
    # ============================ #
    # Penjelasan: Metode ini bertanggung jawab untuk menyimpan model dan tokenizer
    # yang sedang dilatih ke direktori penyimpanan yang ditentukan. Ini memastikan
    # bahwa model yang mencapai performa terbaik selama pelatihan dapat disimpan
    # dan dimuat kembali nanti.
    def _save_model(self) -> None:
        # Buat direktori jika belum ada
        os.makedirs(self.save_dir, exist_ok=True)
        self.tokenizer.save_pretrained(self.save_dir)
        self.model.save_pretrained(self.save_dir)
        print(f"Model disimpan di: {self.save_dir}")

    # ============================ #
    #      Save Loss History       #
    # ============================ #
    # Penjelasan: Metode ini menyimpan riwayat loss pelatihan dan validasi
    # yang terkumpul selama proses pelatihan ke dalam file JSON. File ini
    # dapat digunakan nanti untuk membuat grafik performa model seiring waktu.
    def _save_loss_history(self) -> None:
        history_path = os.path.join(self.save_dir, "loss_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"Riwayat loss disimpan di: {history_path}")