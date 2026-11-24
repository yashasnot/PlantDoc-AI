import pandas as pd
from PIL import Image, ImageDraw
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "reports/subset_15class_predictions.csv")

pairs = [
    ("Tomato___Late_blight", "Tomato___Early_blight"),
    ("Tomato___Target_Spot",
     "Tomato___Spider_mites Two-spotted_spider_mite")
]

W, H = 320, 280
HEADER = 55
images = []

for actual, predicted in pairs:
    errors = df[
        (df.true_class == actual) &
        (df.predicted_class == predicted)
    ].head(5)

    for _, row in errors.iterrows():
        path = Path(row.image_path)

        if not path.exists():
            print("Missing:", path)
            continue

        img = Image.open(path).convert("RGB")
        img.thumbnail((W - 20, H - HEADER - 15))

        images.append((img.copy(), actual, predicted, path.name))

cols = 2
rows = (len(images) + cols - 1) // cols

sheet = Image.new("RGB", (cols * W, rows * H), "white")
draw = ImageDraw.Draw(sheet)

for i, (img, actual, predicted, filename) in enumerate(images):
    x = (i % cols) * W
    y = (i // cols) * H

    draw.text((x + 8, y + 5), f"Actual: {actual.replace('Tomato___', '')}", fill="black")
    draw.text((x + 8, y + 22), f"Predicted: {predicted.replace('Tomato___', '')}", fill="red")

    sheet.paste(img, (x + 10, y + HEADER))
    draw.text((x + 8, y + H - 20), filename[:42], fill="black")

out = ROOT / "reports/misclassified_samples_contact_sheet.jpg"
sheet.save(out, quality=90)

print("Images included:", len(images))
print("Saved:", out)
