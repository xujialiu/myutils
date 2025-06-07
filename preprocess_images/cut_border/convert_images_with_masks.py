import fundus_prep as prep
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":
    src_root_path = Path("/data_A/xujialiu/datasets/DDR_seg/all")
    dst_root_path = Path("/data_A/xujialiu/datasets/DDR_seg/test")

    suffix_image = "jpg"
    suffix_mask = "tif"
    suffix_saved = "png"
    name_mask_folders = ["label/EX", "label/MA", "label/HE", "label/SE"]
    name_image_folder = "image"

    for name_mask_folder in name_mask_folders:
        path_dst_mask_folder = dst_root_path / name_mask_folder
        if not path_dst_mask_folder.exists():
            path_dst_mask_folder.mkdir(parents=True, exist_ok=True)

    path_dst_image_folder = dst_root_path / name_image_folder
    if not path_dst_image_folder.exists():
        path_dst_image_folder.mkdir(parents=True, exist_ok=True)

    list_images = list((src_root_path / name_image_folder).glob(f"**/*.{suffix_image}"))

    for image_path in tqdm(list_images, total=len(list_images)):
        img = prep.imread(image_path)

        stem_img = image_path.stem
        name_mask = f"{stem_img}.{suffix_mask}"

        list_masks = []
        for name_mask_folder in name_mask_folders:
            path_mask = src_root_path / name_mask_folder / name_mask
            mask = prep.imread(path_mask)
            list_masks.append(mask)

        (
            img,
            _,
            _,
            list_masks,
            _,
            _,
            _,
        ) = prep.process_with_masks(img, list_masks)

        dst_img_path = (
            dst_root_path / name_image_folder / f"{image_path.stem}.{suffix_saved}"
        )
        prep.imwrite(dst_img_path, img)

        for name_mask_folder, mask in zip(name_mask_folders, list_masks):
            dst_mask_path = (
                dst_root_path / name_mask_folder / f"{image_path.stem}.{suffix_saved}"
            )
            prep.imwrite(dst_mask_path, mask)
