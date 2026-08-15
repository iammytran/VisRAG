from datasets import load_dataset, Dataset, Image
import shutil
import glob
import os
from pathlib import Path

def main():
    # Bật streaming=True
    corpus_ds = load_dataset("openbmb/VisRAG-Ret-Test-InfoVQA", name="corpus", split="train", streaming=True)

    # Lấy dữ liệu của 1 cột cụ thể
    column_name = "corpus-id"
    images_in_test = []

    for sample in corpus_ds:
        images_in_test.append(sample[column_name].split('.')[0])

    print(f"Đã lấy {len(images_in_test)} dòng.")


    # get layout for test
    for image in images_in_test:
        original_image_layout_dir = f"artifacts/InfoVQA/layout/dev/{image}"
        dest_image_layout_dir = f"artifacts/InfoVQA/layout/test/{image}"
        shutil.copytree(original_image_layout_dir, dest_image_layout_dir, dirs_exist_ok=True)
    print(f"Đã copy {len(images_in_test)} layout directories cho phần test")


    # chuyển các hình ảnh vào sub_images
    sub_images_path = glob.glob("artifacts/InfoVQA/layout/test/**/*.jpg", recursive=True)
    for sub_image_path in sub_images_path:
        filename = os.path.basename(sub_image_path)
        dir_ends_with_image_name = os.path.dirname(sub_image_path) # lấy từ 'artifacts/InfoVQA/layout/test/10022/0_title_0.jpg' sang 'artifacts/InfoVQA/layout/test/10022' 
        folder_image = os.path.basename(dir_ends_with_image_name)

        dest_path = f"artifacts/InfoVQA/image_component_sub/test/{folder_image}_{filename}"
        shutil.copy2(sub_image_path, dest_path)

def create_my_own_test_parquet():
    # 1. Prepare your data with file paths or PIL Image objects
    folder = Path("artifacts/InfoVQA/image_component_sub/test")

    # Dùng sorted() trực tiếp
    sorted_files = sorted(folder.rglob("*.jpg"))
    images = [str(p) for p in sorted_files]
    corpus_ids = [image.name for image in sorted_files]

    print(corpus_ids[:5])
    print(images[:5])

    data = {
        "corpus-id": corpus_ids,
        "image": images,  # Can be list of local file paths or PIL Images
    }

    # 2. Convert to Hugging Face Dataset
    dataset = Dataset.from_dict(data)

    # 3. Cast the image column to the Image feature (reads & encodes bytes automatically)
    dataset = dataset.cast_column("image", Image())

    # 4. Save to Parquet
    dataset.to_parquet("artifacts/InfoVQA/datasets/fragment_corpus.parquet")

    print("Tạo file parquet thành công!")
    return

if __name__ == "__main__":
    main()
    create_my_own_test_parquet()

