
import asyncio
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modifica_datos import obten_media_y_diferencia

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
    
    def predict(self, x) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            if isinstance(x, pd.DataFrame):
                x = x.values.astype(np.float32)
            elif isinstance(x, np.ndarray):
                x = x.astype(np.float32)
            else:
                raise TypeError(f"Tipo no soportado: {type(x)}")

            x_tensor = torch.tensor(x, dtype=torch.float32)
            y_tensor = self.forward(x_tensor)
            return y_tensor.numpy().ravel()
    
async def train_torch_model() -> MLPRegressor:
    df = await obten_media_y_diferencia()

    X = df[["Δcar1²", "media_car1", "Δcar2²", "media_car2", "Δcar3²", "media_car3"]].values.astype("float32")
    y = df["ΔV²"].values.astype("float32")
    eciede2000 = df["ΔECIEDE2000"].values.astype("float32")

    scaler_X = StandardScaler()
    # X_scaled = scaler_X.fit_transform(X)

    scaler_y = StandardScaler()
    # y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    X_train, X_test, y_train, y_test, _, eciede2000_test = train_test_split(
        X, y, eciede2000, test_size=0.3, random_state=42
    )

    # X_train, X_test, y_train, y_test, = train_test_split(
    #     X_scaled, y_scaled, test_size=0.3, random_state=42
    # )

    # Convertimos a tensores
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPRegressor(input_dim=X.shape[1]).to(device)
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

    return model
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



if __name__ == "__main__":
    from train_validate_test.entrenamiento_train_validate_test import evaluar_modelo

    model = asyncio.run(train_torch_model())

    asyncio.run(evaluar_modelo(model))
