from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import colorsys
import matplotlib.pyplot as plt

# --- SETTINGS ---
image_path = "veracityyouth.jpg"  # <-- Replace with your actual image path
num_colors = 5                 # Number of color clusters to identify

# --- LOAD AND PREP IMAGE ---
img = Image.open(image_path).convert("RGB")
img_small = img.resize((50, 50))  # Resize to simplify color clustering
pixels = np.array(img_small).reshape(-1, 3)

# --- K-MEANS COLOR CLUSTERING ---
kmeans = KMeans(n_clusters=num_colors, random_state=0).fit(pixels)
colors = kmeans.cluster_centers_.astype(int)

# --- DISPLAY COLORS FOR SELECTION ---
print("Dominant Colors:")
for idx, color in enumerate(colors):
    print(f"{idx+1}. RGB: {tuple(color)}")

# Optional: show swatches visually
plt.figure(figsize=(10, 2))
for i, color in enumerate(colors):
    plt.subplot(1, num_colors, i+1)
    plt.imshow([[color / 255]])
    plt.axis('off')
plt.tight_layout()
plt.show()

# --- SELECT COLOR ---
choice = int(input(f"Select a color (1-{num_colors}): ")) - 1
selected_rgb = tuple(colors[choice])

# --- CONVERT TO HSB ---
r, g, b = [x / 255.0 for x in selected_rgb]
h, s, v = colorsys.rgb_to_hsv(r, g, b)
hsb = (round(h * 360), round(s * 100), round(v * 100))

# --- CONVERT TO CMYK ---
def rgb_to_cmyk(r, g, b):
    if (r, g, b) == (0, 0, 0):
        return 0, 0, 0, 100
    c = 1 - r / 255
    m = 1 - g / 255
    y = 1 - b / 255
    min_cmy = min(c, m, y)
    c = round((c - min_cmy) / (1 - min_cmy) * 100)
    m = round((m - min_cmy) / (1 - min_cmy) * 100)
    y = round((y - min_cmy) / (1 - min_cmy) * 100)
    k = round(min_cmy * 100)
    return c, m, y, k

cmyk = rgb_to_cmyk(*selected_rgb)

# --- OUTPUT ---
print(f"\nSelected Color (RGB): {selected_rgb}")
print(f"HSB: {hsb}")
print(f"CMYK: {cmyk}")
