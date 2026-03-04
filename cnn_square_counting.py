import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


# e ŞIKKINDA CNN MODELİ KULLANILMISTIR.
# veri üretim fonksiyonu
# bu fonk, 25x25 matris ve bu matrisdeki kare sayısını üretir
def generate_sample_e(
    size=25, min_squares=1, max_squares=10, min_square_size=1, max_square_size=5
):
    mat = np.zeros((size, size), dtype=np.uint8)  # boş bir matris oluşturulur
    num_squares = random.randint(
        min_squares, max_squares
    )  # random bir kare sayısı seçilir
    for _ in range(num_squares):
        square_size = random.randint(
            min_square_size, max_square_size
        )  # random bir kare boyutu seçilir
        x = random.randint(
            0, size - square_size
        )  # kare için random bir x konumu seçilir
        y = random.randint(
            0, size - square_size
        )  # kare için random bir y konumu seçilir
        mat[x : x + square_size, y : y + square_size] = 1  # kare matrise işlenir
    return mat, num_squares  # matris ve kare sayısı döndürülür


# veri seti oluşturma fonksiyonu
# belirtilen sayıda örnek oluşturur
def generate_dataset_e(n_samples, size=25):
    X, y = [], []
    for _ in range(n_samples):
        mat, num_squares = generate_sample_e(size)
        X.append(mat)
        y.append(num_squares)
    return np.array(X), np.array(y)  # matrisler ve kare sayıları döndürülür


# veri oluşturma
X_train, y_train = generate_dataset_e(800)  # eğitim verisi oluşturulur
X_test, y_test = generate_dataset_e(200)  # test verisi oluşturulur

# eğitim oranını kullanıcıdan alma
valid_fractions = [0.25, 0.50, 1.0]  # geçerli eğitim oranları
while True:
    try:
        train_fraction = float(input("Eğitim oranını girin (0.25, 0.50, 1.0): "))
        if train_fraction in valid_fractions:
            break
        else:
            print("Geçersiz değer. Lütfen 0.25, 0.50 veya 1.0 girin.")
    except ValueError:
        print("Geçersiz giriş. Lütfen bir sayı girin.")

# eğitim ve test veri setlerini ayırma
if train_fraction == 1.0:
    train_size = len(X_train) - 1  # doğrulama seti için en az bir örnek ayır
else:
    train_size = int(len(X_train) * train_fraction)
X_train, X_val = X_train[:train_size], X_train[train_size:]
y_train, y_val = y_train[:train_size], y_train[train_size:]

# tensöre çevirme
# numpy dizileri pytorch tensörlerine dönüştürülür
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


# CNN modeli


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # birinci konvolüsyon katmanı
            nn.ReLU(),  # aktivasyon fonksiyonu
            nn.MaxPool2d(2, 2),  # birinci havuzlama katmanı
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # ikinci konvolüsyon katmanı
            nn.ReLU(),  # aktivasyon fonksiyonu
            nn.MaxPool2d(2, 2),  # ikinci havuzlama katmanı
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 6 * 6, 64),  # tam bağlantılı katman
            nn.ReLU(),  # aktivasyon fonksiyonu
            nn.Linear(64, 1),  # çıkış katmanı
        )

    def forward(self, x):
        x = self.conv_layers(x)  # konvolüsyonel katmanlardan geçir
        x = x.view(x.size(0), -1)  # düzleştir
        x = self.fc_layers(x)  # tam bağlantılı katmanlardan geçir
        return x  # modelin tahmin ettiği değer döndürülür


# eğitim ve değerlendirme fonksiyonları
# modeli eğitmek için kullanılır
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()  # gradyanlar sıfırlanır
        outputs = model(X_batch)  # model çıktıları hesaplanır
        loss = criterion(outputs, y_batch)  # kayıp hesaplanır
        loss.backward()  # gradyanlar geriye yayılır
        optimizer.step()  # optimizasyon adımı yapılır
        total_loss += loss.item() * X_batch.size(0)  # toplam kayıp güncellenir
    return total_loss / len(loader.dataset)  # ortalama kayıp döndürülür


# modeli değerlendirmek için kullanılır
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


# modeli eğit
model = CNN()  # CNN modeli oluşturulur
criterion = nn.MSELoss()  # kayıp fonksiyonu olarak ortalama kare hatası kullanılır
optimizer = optim.Adam(
    model.parameters(), lr=0.01
)  # Adam optimizasyon algoritması kullanılır

train_losses = []
test_losses = []

for epoch in range(20):
    train_loss = train(
        model, train_loader, optimizer, criterion
    )  # eğitim kaybı hesaplanır
    test_loss = evaluate(model, val_loader, criterion)  # test kaybı hesaplanır
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print(
        f"Epoch {epoch+1}/20 - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f}"
    )

# eğitim ve test kayıplarını tek grafikte göster
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Eğitim Kaybı")
plt.plot(test_losses, label="Test Kaybı", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Eğitim ve Test Kayıpları")
plt.grid(True)
plt.legend()
plt.show(block=False)


# tahminlerin görselleştirilmesi
# modelin tahminlerini görselleştirir
def visualize_predictions_e(model, X_test, y_test):
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    axs = axs.flatten()

    for i in range(10):
        input_matrix = X_test[i]
        true_count = y_test[i]
        input_tensor = (
            torch.tensor(input_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )
        predicted_count = round(model(input_tensor).item())

        ax = axs[i]

        ax.imshow(input_matrix, cmap="gray", interpolation="none", origin="upper")

        # kare grid çizimi (hücre sınırları için)
        ax.set_xticks(np.arange(-0.5, 25, 1), minor=False)
        ax.set_yticks(np.arange(-0.5, 25, 1), minor=False)
        ax.grid(which="major", color="white", linestyle="-", linewidth=0.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(left=False, bottom=False)
        ax.set_aspect("equal")
        ax.set_xlim(-0.5, 24.5)
        ax.set_ylim(24.5, -0.5)

        ax.set_title(f"Gerçek: {true_count} Tahmin: {predicted_count}", fontsize=8)

    plt.suptitle("Kare Sayısı ve Tahminler", fontsize=14)
    plt.tight_layout()
    plt.show()


visualize_predictions_e(model, X_test, y_test)
