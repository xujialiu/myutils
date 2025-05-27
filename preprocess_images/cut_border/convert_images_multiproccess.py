import fundus_prep as prep
from skimage import io
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from functools import partial

def process_image(image_path, dst_root_path):
    """处理单个图像并保存结果"""
    img = prep.imread(image_path)
    
    # 初始化空列表用于当前处理结果
    current_radius = []
    current_centre_w = []
    current_centre_h = []
    
    # 处理图像（假设每次处理独立，不需要历史数据）
    r_img, borders, mask, _, current_radius, current_centre_w, current_centre_h = prep.process_without_gb(
        img, img, current_radius, current_centre_w, current_centre_h
    )
    
    # 提取当前处理结果
    radius = current_radius[-1] if current_radius else None
    centre_w = current_centre_w[-1] if current_centre_w else None
    centre_h = current_centre_h[-1] if current_centre_h else None
    
    # 保存处理后的图像
    dst_path = Path(dst_root_path) / f"{image_path.stem}.png"
    io.imsave(dst_path, r_img)
    
    return radius, centre_w, centre_h

if __name__ == '__main__':
    src_root = Path("/data_A/xujialiu/datasets/FP_UWF/FP/New")
    dst_root = Path("/data_A/xujialiu/datasets/FP_UWF/FP/FP_all")
    dst_root.mkdir(parents=True, exist_ok=True)
    
    image_paths = list(src_root.glob("**/*.tif"))
    
    # 创建处理函数的部分应用（固定目标路径参数）
    processor = partial(process_image, dst_root_path=dst_root)
    
    # 存储结果的列表
    results = []
    radius_list = []
    centre_list_w = []
    centre_list_h = []
    
    # 创建进程池并行处理
    with multiprocessing.Pool(processes=10) as pool:
        # 使用imap保持处理顺序（如果需要）
        for result in tqdm(pool.imap(processor, image_paths), total=len(image_paths)):
            results.append(result)
    
    # 解包所有结果
    for r, w, h in results:
        if r is not None and w is not None and h is not None:
            radius_list.append(r)
            centre_list_w.append(w)
            centre_list_h.append(h)