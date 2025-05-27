import fundus_prep as prep
from skimage import io
from pathlib import Path
from tqdm import tqdm


if __name__ == '__main__':
    src_root_path = Path("/data_A/xujialiu/datasets/FP_UWF/FP/New")
    dst_root_path = Path("/data_A/xujialiu/datasets/FP_UWF/FP/FP_all")
    
    if not dst_root_path.exists():
        dst_root_path.mkdir(parents=True)
    
    list_images = list(src_root_path.glob("**/*.tif"))
    radius_list = []    
    centre_list_w = []
    centre_list_h = []
    for image_path in tqdm(list_images, total=len(list_images)):
        img = prep.imread(
            image_path
        )
        r_img, borders, mask, r_img, radius_list, centre_list_w, centre_list_h = (
            prep.process_without_gb(img, img, radius_list, centre_list_w, centre_list_h)
        )
        io.imsave(dst_root_path / (image_path.stem + ".png"), r_img)

