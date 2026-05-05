import sys
import os

# Явно добавляем рабочую папку и папку app в пути Python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
app_path = os.path.join(BASE_DIR, 'app')
if os.path.isdir(app_path):
    sys.path.insert(0, app_path)

import torch
import numpy as np
import pickle
from services.stgcn_service import stgcn_runner

def debug_stgcn():
    print("=== ДИАГНОСТИКА STGCN ===")
    
    # 1. Проверяем матрицу смежности
    adj = stgcn_runner.adj_tensor
    print(f"1. Форма матрицы смежности: {adj.shape} (Должна быть [24, 24])")
    # Считаем изолированные узлы (сумма связей по строке равна 0)
    print(f"   Изолированных станций: {(adj.sum(dim=1) == 0).sum().item()}")
    
    # 2. Генерируем тестовый тензор (Имитация реального потока)
    dummy_input = torch.rand((1, 12, 24, 6), dtype=torch.float32)
    
    # Подаем тензор в модель
    stgcn_runner.model.eval()
    with torch.no_grad():
        x_gcn = stgcn_runner.model.gcn(dummy_input, adj)
        print(f"2. Выход после GCN (Пространство): {x_gcn.shape}")
        
        raw_pred = stgcn_runner.model(dummy_input, adj)
        print(f"3. Выходной прогноз модели (Сырой тензор): {raw_pred.shape}")
        
    # 3. Проверка скейлера
    with open(stgcn_runner.scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    print(f"4. Scaler тип: {type(scaler)}")
    try:
        pred_numpy = raw_pred.numpy().reshape(-1, 1)
        final_pm25 = scaler.inverse_transform(pred_numpy)
        print(f"5. Обратное масштабирование прошло успешно. Диапазон прогноза: {final_pm25.min():.1f} - {final_pm25.max():.1f}")
    except Exception as e:
        print(f"ОШИБКА СО СКЕЙЛЕРОМ: {e}")

if __name__ == "__main__":
    debug_stgcn()