import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd


# A ŞIKKINDA MLP MODELİ KULLANILMIŞTIR. GERİ KALAN ŞIKLARDA CNN.
# HER ÖRNEKTE TAM HALLERİ İÇİN 800 EĞİTİM,200 TEST KÜMESİ OLUŞTURULMUŞTUR.
# veri uretim fonk
# bu fonk, 25x25 matris ve bu matristeki iki nokta arasındaki mesafeyi üretir
def generate_sample_a(size=25, min_points=3, max_points=10):
    mat = np.zeros((size, size), dtype=np.uint8)  # boş matris oluşturur
    num_points = 2  # sadece iki nokta seçilir
    points = random.sample(
        [(i, j) for i in range(size) for j in range(size)], num_points
    )  # matris üzerinde iki random nokta seçilir
    for x, y in points:
        mat[x, y] = 1  # seçilen noktalar matrise işlenir
    point1, point2 = points
    distance = np.sqrt(
        (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
    )  # iki nokta arasındaki mesafe hesaplanir
    return mat, distance


# veri seti oluşturma fonk
# belirtilen sayıda örnek oluşturur
def generate_dataset_a(n_samples, size=25):
    X, y = [], []
    for _ in range(n_samples):
        mat, max_dist = generate_sample_a(size)
        X.append(mat)
        y.append(max_dist)
    return np.array(X), np.array(y)  # matrisler ve mesafeler döndürülür


# veri olusturma. egitim icin 800 test icin  200 ornek. (tam halleri icin)
X_train, y_train = generate_dataset_a(800)
X_test, y_test = generate_dataset_a(200)

# veri setini kaydetme(dataset)
# np.save("X_train.npy", X_train)
# np.save("y_train.npy", y_train)
# np.save("X_test.npy", X_test)
# np.save("y_test.npy", y_test)

# tensore çevirme
# numpy dizileri pytorch tensörlerine dönüştürülür
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


#  MLP modeli
# bu model giriş verilerini alır ve birkaç katman boyunca işleyerek bir çıktı üretir
# MLP tam bağlantılı katmanlardan oluşur ve her bir nöron, bir önceki katmandaki tüm nöronlarla bağlantılıdır
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(
                25 * 25, 128
            ),  # giriş katmanı: 25x25 boyutundaki matris düzleştirilir ve 128 nöronlu bir katmana bağlanır
            nn.ReLU(),  # aktivasyon fonksiyonu: ReLU, doğrusal olmayan bir aktivasyon fonksiyonudur ve negatif değerleri sıfırlar
            nn.Linear(128, 64),  # gizli katman: 128 nörondan 64 nörona geçiş yapılır.
            nn.ReLU(),  # ikinci aktivasyon fonksiyonu
            nn.Linear(
                64, 1
            ),  # cıkıs katmanı: 64 nörondan tek bir çıkış değerine geçiş yapılır.
        )

    def forward(self, x):
        x = x.view(
            x.size(0), -1
        )  # giris verisi düzleştirilir: 2D matris, 1D vektöre dönüştürülür
        x = self.layers(x)  # giriş verisi, tanımlanan katmanlar boyunca işlenir
        return x  # modelin tahmin ettiği değer döndürülür


# Eğitim ve değerlendirme fonksiyonları
# modeli eğitmek için kullanılır.
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()  # gradyanlar sıfırlanır
        outputs = model(X_batch)  # model çıktıları hesaplanır
        loss = criterion(outputs, y_batch)  # loss hesaplanır
        loss.backward()  # gradyanlar geriye yayılır
        optimizer.step()  # Optimizasyon adımı yapılır
        total_loss += loss.item() * X_batch.size(0)  # total loss güncellenir
    return total_loss / len(loader.dataset)  # ort. kayıp döndürülür


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


# Eğitim veri oranı karşılaştırması
def get_training_fractions():
    valid_fractions = {0.25, 0.50, 1.0}
    while True:
        fractions = input("Eğitim oranını girin (örnek: 0.25, 0.5, 1.0): ")
        try:
            fractions_list = [float(f.strip()) for f in fractions.split(",")]
            if all(f in valid_fractions for f in fractions_list):
                return fractions_list
            else:
                print("Lütfen sadece 0.25, 0.50 ve 1.0 değerlerini girin.")
        except ValueError:
            print("Geçersiz giriş. Lütfen tekrar deneyin.")


def main():
    # kullanıcıdan eğitim oranını alır
    fractions = get_training_fractions()
    results = {}

    # eğitim oranı karşılaştırması
    for frac in fractions:
        n_samples = int(frac * len(X_train_tensor))
        subset_X = X_train_tensor[:n_samples]
        subset_y = y_train_tensor[:n_samples]

        dataset = TensorDataset(subset_X, subset_y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        model_frac = MLP()
        optimizer_frac = optim.Adam(model_frac.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        for epoch in range(10):
            train(model_frac, loader, optimizer_frac, criterion)

        test_loss = evaluate(model_frac, test_loader, criterion)
        results[f"{int(frac*100)}% Eğitim"] = test_loss

    # eğitim ve test kayıplarını hesapla
    model = MLP()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    test_losses = []

    for epoch in range(20):
        train_loss = train(model, train_loader, optimizer, criterion)
        test_loss = evaluate(model, test_loader, criterion)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(
            f"Epoch {epoch+1}/20 - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f}"
        )
    # gorsellestirme aşamaları:
    # Eğitim ve test kayıplarını tek grafikte gösterir
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Eğitim Kaybı")
    plt.plot(test_losses, label="Test Kaybı", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Eğitim ve Test Kayıpları")
    plt.grid(True)
    plt.legend()
    plt.show(block=False)

    visualize_predictions(model)


def visualize_predictions(model):
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    axs = axs.flatten()

    for i in range(10):
        input_matrix = X_test[i]
        true_distance = y_test[i]
        input_tensor = (
            torch.tensor(input_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )
        predicted_distance = model(input_tensor).item()

        points = np.argwhere(input_matrix == 1)
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

        # nokta konumları
        for point in points:
            ax.scatter(point[1], point[0], color="red", s=40, edgecolors="white")

        # en uzak iki noktayı bul ve çiz
        max_dist = 0
        farthest_pair = (None, None)
        for j in range(len(points)):
            for k in range(j + 1, len(points)):
                dist = np.sqrt(
                    (points[j][0] - points[k][0]) ** 2
                    + (points[j][1] - points[k][1]) ** 2
                )
                if dist > max_dist:
                    max_dist = dist
                    farthest_pair = (points[j], points[k])

        if farthest_pair[0] is not None and farthest_pair[1] is not None:
            ax.plot(
                [farthest_pair[0][1], farthest_pair[1][1]],
                [farthest_pair[0][0], farthest_pair[1][0]],
                color="yellow",
                linestyle="--",
            )

        ax.set_title(
            f"Gerçek: {true_distance:.2f}\nTahmin: {predicted_distance:.2f}", fontsize=8
        )

    plt.suptitle("Nokta Konumları ve Tahminler", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
