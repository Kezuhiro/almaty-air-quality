import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

print("--- ЭТАП 3.2: ОБУЧЕНИЕ STGCN ---")

# 1. Загружаем Матрицу Смежности (A)
adj_matrix = np.load("adj_matrix.npy")
# Переводим в тензор PyTorch
adj_tensor = torch.FloatTensor(adj_matrix)

# 2. Архитектура Графового Слоя (Spatial Block)
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Веса для трансформации фичей
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        # x: (Batch, Seq, Nodes, Features)
        # Умножаем фичи на веса
        support = torch.matmul(x, self.weight) 
        
        # МАГИЯ ГРАФОВ: Умножаем на матрицу смежности
        # Соседи обмениваются информацией друг с другом пропорционально весам связей
        out = torch.einsum('vw,bswf->bsvf', adj, support)
        return F.relu(out)

# 3. Полная Архитектура STGCN
class AlmatySTGCN(nn.Module):
    def __init__(self, num_nodes, num_features, hidden_dim, seq_len):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        
        # Шаг 1: Графовая свертка (извлекаем пространственные связи)
        self.gcn = GraphConvLayer(in_features=num_features, out_features=hidden_dim)
        
        # Шаг 2: Временной блок (GRU для анализа последовательности)
        # Схлопываем узлы и фичи, чтобы передать во временную сеть
        self.gru = nn.GRU(input_size=num_nodes * hidden_dim, 
                          hidden_size=hidden_dim * 2, 
                          batch_first=True)
        
        # Шаг 3: Полносвязный слой (Выдаем прогноз PM2.5 для каждой из 24 станций)
        self.fc = nn.Linear(hidden_dim * 2, num_nodes)

    def forward(self, x, adj):
        batch_size = x.size(0)
        
        # Пропускаем через графовый слой
        x_gcn = self.gcn(x, adj) # -> [Batch, Seq, Nodes, Hidden]
        
        # Подготавливаем для GRU
        x_gru_in = x_gcn.reshape(batch_size, self.seq_len, -1)
        
        # Пропускаем через GRU
        gru_out, _ = self.gru(x_gru_in) # -> [Batch, Seq, Hidden*2]
        
        # Берем только последний шаг по времени (нам важен финал истории)
        last_step = gru_out[:, -1, :] 
        
        # Делаем прогноз
        out = self.fc(last_step) # -> [Batch, 24]
        return out

# 4. Инициализация Модели
NUM_NODES = 24
NUM_FEATURES = 6
HIDDEN_DIM = 16
SEQ_LEN = 12

model = AlmatySTGCN(num_nodes=NUM_NODES, num_features=NUM_FEATURES, 
                    hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN)

# Выбираем оптимизатор (Adam) и функцию потерь (MSE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Проверяем, есть ли видеокарта (GPU) для ускорения
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
adj_tensor = adj_tensor.to(device)
print(f"Обучение запущено на: {device}")

# 5. ЦИКЛ ОБУЧЕНИЯ (Training Loop)
EPOCHS = 30

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    
    # Идем по батчам обучающей выборки
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad() # Обнуляем градиенты
        
        # Делаем прогноз
        predictions = model(x_batch, adj_tensor)
        
        # Считаем ошибку
        loss = criterion(predictions, y_batch)
        
        # Шаг назад (Backpropagation)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * x_batch.size(0)
        
    train_loss /= len(train_loader.dataset)
    
    # --- ВАЛИДАЦИЯ ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            val_preds = model(x_val, adj_tensor)
            loss = criterion(val_preds, y_val)
            val_loss += loss.item() * x_val.size(0)
            
    val_loss /= len(val_loader.dataset)
    
    # Печатаем лог каждые 5 эпох
    if epoch % 5 == 0 or epoch == 1:
        print(f"Эпоха {epoch:02d}/{EPOCHS} | Ошибка Train: {train_loss:.5f} | Ошибка Val: {val_loss:.5f}")

# Сохраняем веса обученной нейросети
torch.save(model.state_dict(), "stgcn_almaty.pth")
print("\nМодель обучена и сохранена в файл stgcn_almaty.pth!")