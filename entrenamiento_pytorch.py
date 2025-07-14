import asyncio
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

from graficas import graficar_regresión_lineal_dataframe
from modifica_datos import obten_media_y_diferencia

def stress_torch(y_true, y_pred):
    numerator_F1 = torch.sum(y_pred**2)
    denominator_F1 = torch.sum(y_pred * y_true)
    f1 = numerator_F1 / denominator_F1

    numerator = torch.sum((y_pred - f1 * y_true)**2)
    denominator = torch.sum((f1 * y_true)**2)

    stress = torch.sqrt(numerator / denominator)
    return stress

class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 110),
            nn.ReLU(),
            nn.Linear(110, 116),
            nn.ReLU(),
            nn.Linear(116, 121),
            nn.ReLU(),
            nn.Linear(121, 1),
        )

    def forward(self, x):
        return self.model(x)

async def main():
    df = await obten_media_y_diferencia()

    X = df[["Δcar1²", "media_car1", "Δcar2²", "media_car2", "Δcar3²", "media_car3"]].values.astype("float32")
    y = df["ΔV²"].values.astype("float32")
    eciede2000 = df["ΔECIEDE2000"].values.astype("float32")

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    X_train, X_test, y_train, y_test, _, eciede2000_test = train_test_split(
        X_scaled, y_scaled, eciede2000, test_size=0.3, random_state=42
    )

    # X_train, X_test, y_train, y_test, = train_test_split(
    #     X_scaled, y_scaled, test_size=0.3, random_state=42
    # )

    # Convertimos a tensores
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

    model = MLPRegressor(input_dim=X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.006087283490459225)#lr=0.001)
    loss_fn = nn.MSELoss()

    batch_size = 64
    n_epochs = 100
    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Entrenamiento
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in dataloader:
            optimizer.zero_grad()
            output = model(xb)
            loss = loss_fn(output, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        # output = model(X_train_t)
        # loss = loss_fn(output, y_train_t)
        # loss.backward()
        # optimizer.step()
        if epoch % 10 == 0:
            with torch.no_grad():
                val_pred = model(X_test_t)
                val_loss = loss_fn(val_pred, y_test_t)
                # print(f"Epoch {epoch} - Train Loss: {loss.item():.4f} - Val Loss: {val_loss.item():.4f}")
            avg_loss = total_loss / len(dataloader.dataset)
            print(f"Epoch {epoch} - Train Loss: {avg_loss:.4f} - Val Loss: {val_loss.item():.4f}")

    # Predicción y evaluación
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_t).numpy().ravel()

    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_test_orig = scaler_y.inverse_transform(y_test_t.numpy().reshape(-1, 1)).ravel()

    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
    print(f"RMSE: {rmse:.4f}")

    #TODO: Calcular el stress de ECIEDE2000 del 30% de los datos de entreneamiento
    # Stress (usamos tensores directamente)
    stress = stress_torch(
        torch.tensor(y_test_orig, dtype=torch.float32),
        torch.tensor(y_pred, dtype=torch.float32)
    ).item()
    print(f"Stress: {stress:.4f}")

    # DataFrame para graficar
    label_df = pd.DataFrame({
        "ΔV": np.sqrt(y_test_orig),
        "Δpred": np.sqrt(y_pred),
        "ΔECIEDE2000": eciede2000_test
    })
    await graficar_regresión_lineal_dataframe(label_df, "ΔV", "Δpred", "ΔECIEDE2000")

asyncio.run(main())
