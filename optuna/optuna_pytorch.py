import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import optuna
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modifica_datos import obten_media_y_diferencia

df = None
X_train = None
X_test = None
delta_V_train = None
delta_V_test = None
input = None
label = None

async def obten_datos():
    global df
    df = await obten_media_y_diferencia()

    # input = df.drop(columns=["ΔV²", "ΔECIEDE2000"])
    # label = df["ΔV²"]

def build_model(trial, input_dim):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []

    in_features = input_dim
    for i in range(n_layers):
        out_features = trial.suggest_int(f"n_units_l{i}", 16, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        in_features = out_features

    layers.append(nn.Linear(in_features, 1))
    return nn.Sequential(*layers)

def stress_torch(y_true, y_pred):
    numerator_F1 = torch.sum(y_pred**2)
    denominator_F1 = torch.sum(y_pred * y_true)
    f1 = numerator_F1 / denominator_F1
    numerator = torch.sum((y_pred - f1 * y_true)**2)
    denominator = torch.sum((f1 * y_true)**2)
    return torch.sqrt(numerator / denominator)

def objective(trial):
    # Cargar y preparar los datos
    # df = asyncio.run(obten_media_y_diferencia())

    X = df[["Δcar1²", "media_car1", "Δcar2²", "media_car2", "Δcar3²", "media_car3"]].values.astype("float32")
    y = df["ΔV²"].values.astype("float32")

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)

    model = build_model(trial, input_dim=X.shape[1])

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

    loss_fn = nn.MSELoss()
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    num_epochs = 50

    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluación
    model.eval()
    with torch.no_grad():
        y_pred_val_scaled = model(X_val_t).squeeze()
    
    y_pred_val = scaler_y.inverse_transform(y_pred_val_scaled.numpy().reshape(-1, 1)).ravel()
    y_val_orig = scaler_y.inverse_transform(y_val.reshape(-1, 1)).ravel()

    stress = stress_torch(
        torch.tensor(y_val_orig, dtype=torch.float32),
        torch.tensor(y_pred_val, dtype=torch.float32)
    ).item()

    return stress

    # model.eval()
    # with torch.no_grad():
    #     y_pred_val = model(X_val_t).numpy().ravel()
    # y_pred_val = scaler_y.inverse_transform(y_pred_val.reshape(-1, 1)).ravel()
    # y_val_orig = scaler_y.inverse_transform(y_val.reshape(-1, 1)).ravel()

    # rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred_val))
    # return rmse

async def main():
    # Cargar los datos
    await obten_datos()
    # Ejecutar Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("Mejores hiperparámetros:", study.best_params)
    print("Mejor RMSE:", study.best_value)

if __name__ == "__main__":
    asyncio.run(main())

