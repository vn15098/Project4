import pickle
from pathlib import Path

from featureExtraction import extract_features_batch

project_dir = Path(__file__).resolve().parent
image_folder = project_dir / "images"
artifacts_dir = project_dir / "artifacts"
artifacts_dir.mkdir(exist_ok=True)

if not image_folder.is_dir():
    raise FileNotFoundError(f"Image directory does not exist: {image_folder}")

filenames = sorted(
    path for path in image_folder.iterdir()
    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
)
if not filenames:
    raise ValueError(f"No JPG or PNG images found in {image_folder}.")

embeddings = []
batch_size = 32

for start in range(0, len(filenames), batch_size):
    batch_paths = filenames[start:start + batch_size]
    embeddings.extend(extract_features_batch(batch_paths))
    print(f"Processed {min(start + batch_size, len(filenames))}/{len(filenames)} images")

with (artifacts_dir / "embeddings.pkl").open("wb") as output:
    pickle.dump(embeddings, output)

with (artifacts_dir / "filenames.pkl").open("wb") as output:
    pickle.dump([str(path) for path in filenames], output)

print("Embeddings Saved")