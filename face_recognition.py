import os
import numpy as np
from PIL import Image

import kagglehub

# Download latest version
data_path = kagglehub.dataset_download("kasikrit/att-database-of-faces")

D = []
y = []

for subject_id in range(1, 41):
    subject_folder = os.path.join(data_path, f"s{subject_id}")
    for img_name in sorted(os.listdir(subject_folder)):
        img = Image.open(os.path.join(subject_folder, img_name))
        D.append(np.array(img).flatten())  # 92x112 = 10304
        y.append(subject_id)

D = np.array(D)  # shape: (400, 10304)
y = np.array(y)  # shape: (400,)

# Step 2: Split — odd rows (0-indexed: 0,2,4,...) for train, even rows (1,3,5,...) for test
X_train = D[0::2]   # rows 1,3,5,7,9 per subject → 200 samples
X_test  = D[1::2]   # rows 2,4,6,8,10 per subject → 200 samples
y_train = y[0::2]
y_test  = y[1::2]

np.savez("face_data.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
print("Saved to face_data.npz")

print(f"D shape:       {D.shape}")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")
