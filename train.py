import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# 1. Tự động tải file heart.csv từ Internet nếu chưa có
DATA_URL = "https://raw.githubusercontent.com/Ankit152/Heart-Disease-Prediction/master/heart.csv"

if not os.path.exists('heart.csv'):
    print("⏳ Đang tải dữ liệu heart.csv từ internet...")
    df = pd.read_csv(DATA_URL)
    df.to_csv('heart.csv', index=False)
    print("✅ Đã tải xong tệp heart.csv!")
else:
    print("📂 Đã tìm thấy tệp heart.csv sẵn có.")
    df = pd.read_csv('heart.csv')

# 2. Phân chia dữ liệu
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Xây dựng mô hình Keras
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Lưu lại mô hình có độ chính xác cao nhất
checkpoint = ModelCheckpoint(
    'best_model.h5', 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)

# 5. Bắt đầu huấn luyện (100 epoch)
print("🚀 Đang huấn luyện mô hình Keras...")
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[checkpoint]
)

print("\n🎉 THÀNH CÔNG! Tệp 'best_model.h5' đã được tạo trong thư mục dự án!")