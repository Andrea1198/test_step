import numpy as np
import matplotlib.pyplot as plt

mat1 = "./results/HAI_old.npy"
mat2 = "./results/h_aim_ik0.npy"

# mat1 = np.loadtxt(mat1).real[:50,:50]
# mat2 = np.loadtxt(mat2).real[:50,:50]
mat1 = np.loadtxt(mat1).real[:,:]
mat2 = np.loadtxt(mat2).real[:,:]

print(np.max(np.abs(mat1 - mat2)))
print(np.where(np.abs(mat1 - mat2) > 1e-5))

fig, axs = plt.subplots(1, 3, figsize=(12, 6))
axs[0].imshow(mat1, cmap='viridis')
axs[0].set_title('Matrix 1')
axs[0].axis('off')
axs[1].imshow(mat2, cmap='viridis')
axs[1].set_title('Matrix 2')
axs[1].axis('off')
diff = np.abs(mat1 - mat2)
im = axs[2].imshow(diff, cmap='viridis')
axs[2].set_title('Absolute Difference')
axs[2].axis('off')
fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

