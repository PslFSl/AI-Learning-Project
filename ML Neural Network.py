import numpy as np
import matplotlib.pyplot as plt




# ============================================================
# 1. VERİ - XOR PROBLEMİ
# ============================================================

X = np.array([[0,0], [0,1], [1,0], [1,1]])
# 4 satır, 2 sütun -- 4 girdi durumu, her biri 2 özellikli (x1,x2)

y = np.array([[0], [1], [1], [0]])

# ============================================================
# 2. SIGMOID FONKSİYONU VE TÜREVİ
# ============================================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_turev(z):
    s = sigmoid(z)
    return s * (1 - s)



# ============================================================
# 3. AĞIRLIKLARI RASTGELE BAŞLATMA
# ============================================================

np.random.seed(1)
# tekrarlanabilirlik için

# Gizli katman: 2 girdi -> 2 gizli nöron
W1 = np.random.randn(2, 2)

b1 = np.zeros((1, 2))

W2 = np.random.randn(2, 1)

b2 = np.zeros((1, 1))

learning_rate = 1.0

epochs = 10000


# ============================================================
# 4. EĞİTİM DÖNGÜSÜ - FORWARD PASS + BACKPROPAGATION
# ============================================================

loss_gecmisi = []
# her epoch'taki hatayı kaydetmek için

for epoch in range(epochs):

    # ---------- FORWARD PASS (İleri Yayılım) ----------

    z1 = X @ W1 + b1

    a1 = sigmoid(z1)

    z2 = a1 @ W2 + b2

    a2 = sigmoid(z2)

    # ---------- LOSS HESAPLAMA ----------

    loss = np.mean((a2 - y) ** 2)

    loss_gecmisi.append(loss)


    # --- ADIM 1: Çıktı katmanının hatası ---
    dLoss_da2 = 2 * (a2 - y) / len(X)

    da2_dz2 = sigmoid_turev(z2)

    dz2 = dLoss_da2 * da2_dz2

    # --- ADIM 2: Çıktı katmanı ağırlıklarının gradyanı ---
    dW2 = a1.T @ dz2

    db2 = np.sum(dz2, axis=0, keepdims=True)

    # --- ADIM 3: Hatayı gizli katmana GERİ YAY ---
    da1 = dz2 @ W2.T

    dz1 = da1 * sigmoid_turev(z1)

    # --- ADIM 4: Gizli katman ağırlıklarının gradyanı ---
    dW1 = X.T @ dz1

    db1 = np.sum(dz1, axis=0, keepdims=True)
    # gizli katmanın bias gradyanı

    # ---------- GÜNCELLEME (Gradient Descent Adımı) ----------

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")


# ============================================================
# 5. SONUÇLARI TEST ETME
# ============================================================

print("\n--- Eğitilmiş MLP ile XOR Testi ---")
z1_final = X @ W1 + b1
a1_final = sigmoid(z1_final)
z2_final = a1_final @ W2 + b2
a2_final = sigmoid(z2_final)
# eğitim bittikten sonra son bir forward pass daha yapıyoruz, final tahminleri almak için

for i in range(len(X)):
    tahmin_olasilik = a2_final[i][0]
    tahmin_sinif = 1 if tahmin_olasilik > 0.5 else 0
    print(f"Girdi: {X[i]} -> Olasılık: {tahmin_olasilik:.4f} -> "
          f"Sınıf: {tahmin_sinif}, Gerçek: {y[i][0]}")



# ============================================================
# 6. GÖRSELLEŞTİRME - LOSS CURVE
# ============================================================

plt.plot(loss_gecmisi)
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("MLP - XOR Öğrenme Süreci")
plt.show()