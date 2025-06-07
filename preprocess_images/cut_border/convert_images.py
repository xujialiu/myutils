import fundus_prep as prep
from pathlib import Path
from tqdm import tqdm


if __name__ == "__main__":
    src_root_path = Path("/data_A/xujialiu/datasets/DDR_seg/preprocess_single")
    dst_root_path = Path("/data_A/xujialiu/datasets/DDR_seg/test")

    suffix_image = "jpg"
    suffix_saved = "png"

    if not dst_root_path.exists():
        dst_root_path.mkdir(parents=True)

    list_images = list(src_root_path.glob(f"**/*.{suffix_image}"))

    for image_path in tqdm(list_images, total=len(list_images)):
        img = prep.imread(image_path)
        r_img, *_ = prep.process(img)
        prep.imwrite(dst_root_path / f"{image_path.stem}.{suffix_saved}", r_img)