import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


# B ŞIKKINDA CNN MODELİ KULLANILMISTIR.
# veri üretim fonksiyonu
# bu fonk, 25x25 matris ve bu matristeki en yakın iki nokta arasındaki mesafeyi üretir
def generate_sample_b(size=25, min_points=3, max_points=10):
    mat = np.zeros((size, size), dtype=np.uint8)  # boş matris oluşturulur
    num_points = random.randint(
        min_points, max_points
    )  # rastgele bir nokta sayısı seçilir
    points = random.sample(
        [(i, j) for i in range(size) for j in range(size)], num_points
    )  # matris üzerinde rastgele noktalar seçilir
    for x, y in points:
        mat[x, y] = 1  # seçilen noktalar matrise işlenir
    min_distance = float("inf")
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.sqrt(
                (points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2
            )  # iki nokta arasındaki mesafe hesaplanır
            if dist < min_distance:
                min_distance = dist  # en yakın mesafe güncellenir
    return mat, min_distance  # matris ve en yakın mesafe döndürülür


# veri seti oluşturma fonksiyonu
# belirtilen sayıda örnek oluşturur
def generate_dataset_b(n_samples, size=25):
    X, y = [], []
    for _ in range(n_samples):
        mat, min_dist = generate_sample_b(size)
        X.append(mat)
        y.append(min_dist)
    return np.array(X), np.array(y)  # matrisler ve mesafeler döndürülür


# veri olusturma. egitim icin 800 test icin  200 ornek. (tam halleri icin)
X_train, y_train = generate_dataset_b(800)  # eğitim verisi oluşturulur
X_test, y_test = generate_dataset_b(200)  # test verisi oluşturulur

# Eğitim oranını kullanıcıdan alma ve doğrulama
valid_fractions = [0.25, 0.50, 1.0]
train_fraction = None
while train_fraction not in valid_fractions:
    try:
        train_fraction = float(input("Eğitim oranini girin (0.25, 0.50, 1.0): "))
        if train_fraction not in valid_fractions:
            print(
                "Geçersiz oran. Lütfen 0.25, 0.50 veya 1.0 değerlerinden birini girin."
            )
    except ValueError:
        print("Geçersiz giriş. Lütfen bir sayi girin.")

# Eğitim ve test veri setlerini oranlara göre ayırma
train_size = int(len(X_train) * train_fraction)
X_train, y_train = X_train[:train_size], y_train[:train_size]

# tensöre çevirme
# NumPy dizileri PyTorch tensörlerine dönüştürülür
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# CNN (konvolüsyonel sinir ağı)  modeli tanımlanır


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                1, 16, kernel_size=3, padding=1
            ),  # birinci konvolüsyon katmanı ->> 1 giriş kanalı, 16 çıkış kanalı, 3x3 filtre boyutu
            nn.ReLU(),  # aktivasyon fonk-> ReLU, doğrusal olmayan bir aktivasyon fonksiyonudur ve negatif değerleri sıfırlar
            nn.MaxPool2d(
                2, 2
            ),  # birinci havuzlama katmanı -> 2x2 boyutunda havuzlama uygular
            nn.Conv2d(
                16, 32, kernel_size=3, padding=1
            ),  # ikinci konvolüsyon katmanı -> 16 giriş kanalı, 32 çıkış kanalı
            nn.ReLU(),  # aktivasyno fonk
            nn.MaxPool2d(2, 2),  # ikinci havuzlama katmanı
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(
                32 * 6 * 6, 64
            ),  # tam bağlantılı katman: konvolüsyon katmanlarından gelen veriyi 64 nöronlu bir katmana bağlar
            nn.ReLU(),  # aktivasyon fonk
            nn.Linear(
                64, 1
            ),  # cıkıs katmanı -> 64 nörondan tek bir çıkış değerine geçiş yapılır
        )

    def forward(self, x):
        x = self.conv_layers(x)  # konvolüsyonel katmanlardan geçir
        x = x.view(x.size(0), -1)  # düzleştir -> 2D veriyi 1D vektöre dönüştür
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
criterion = nn.MSELoss()  # loss fonksiyonu olarak ortalama kare hatası kullanılır
optimizer = optim.Adam(
    model.parameters(), lr=0.01
)  # Adam optimizasyon algoritması kullanılır

train_losses = []
test_losses = []

for epoch in range(20):
    train_loss = train(
        model, train_loader, optimizer, criterion
    )  # eğitim kaybı hesaplanır
    test_loss = evaluate(model, test_loader, criterion)  # test kaybı hesaplanır
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print(
        f"Epoch {epoch+1}/20 - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f}"
    )

# eğitim ve test lossları tek grafikte gösterir
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
def visualize_predictions_b(model, X_test, y_test):
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

        # en yakın iki noktayı bul ve çiz
        min_dist = float("inf")
        closest_pair = (None, None)
        for j in range(len(points)):
            for k in range(j + 1, len(points)):
                dist = np.sqrt(
                    (points[j][0] - points[k][0]) ** 2
                    + (points[j][1] - points[k][1]) ** 2
                )
                if dist < min_dist:
                    min_dist = dist
                    closest_pair = (points[j], points[k])

        if closest_pair[0] is not None and closest_pair[1] is not None:
            ax.plot(
                [closest_pair[0][1], closest_pair[1][1]],
                [closest_pair[0][0], closest_pair[1][0]],
                color="yellow",
                linestyle="--",
            )

        ax.set_title(
            f"Gerçek: {true_distance:.2f}\nTahmin: {predicted_distance:.2f}", fontsize=8
        )

    plt.suptitle("Nokta Konumları ve Tahminler", fontsize=14)
    plt.tight_layout()
    plt.show()


visualize_predictions_b(model, X_test, y_test)
