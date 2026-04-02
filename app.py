import gradio as gr
import os
import sys 
import subprocess
import glob
import shutil
import numpy as np
import zipfile
from PIL import Image

# def run_bash_task_1(input_image):
#     """
#     Giả lập chức năng 1: Augment
#     Input: Ảnh thật (dạng numpy array)
#     Output: List chứa các ảnh (để hiển thị lên Gallery)
#     """
#     if input_image is None:
#         return []
    
#     print("Đang chạy Task 1 (Giả lập)...")
    
#     # Tạo ra 4 ảnh giả lập kết quả (ở đây mình lấy chính ảnh gốc làm mờ đi để demo)
#     # Thực tế chỗ này sẽ là code gọi file bash
#     results = []
#     for i in range(4):
#         # Giả lập xử lý ảnh: lật ngược ảnh hoặc làm nhiễu để thấy khác biệt
#         fake_result = np.fliplr(input_image) if i % 2 == 0 else input_image
#         results.append(fake_result)
        
#     return results

# def run_bash_task_2():
#     """
#     Giả lập chức năng 2: Synthetic
#     Input: Không có
#     Output: 1 Ảnh giả (dạng numpy array)
#     """
#     print("Đang chạy Task 2 (Giả lập)...")
    
#     # Tạo ra một ảnh nhiễu ngẫu nhiên (noise) kích thước 512x512
#     # Thực tế chỗ này code gọi bash sinh ảnh và đọc ảnh đó lên
#     random_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
#     return random_image

# with gr.Blocks() as demo:
#     gr.Markdown("## Chạy Diff-Palm qua Bash Script (Giao diện Demo)")
    
#     with gr.Tab("Mode 1: Augment"):
#         gr.Markdown("Upload 1 ảnh thật, hệ thống sẽ sinh ra các biến thể.")
#         with gr.Row():
#             with gr.Column():
#                 inp = gr.Image(label="Ảnh đầu vào")
#                 btn1 = gr.Button("Chạy sinh ra thêm ảnh từ ảnh thật", variant="primary")
#             with gr.Column():
#                 out1 = gr.Gallery(label="Kết quả sinh ra", columns=2)
        
#         btn1.click(run_bash_task_1, inputs=inp, outputs=out1)
        
#     with gr.Tab("Mode 2: Synthetic"):
#         gr.Markdown("Hệ thống tự động sinh nếp nhăn và tạo ảnh bàn tay giả.")
#         with gr.Row():
#             with gr.Column():
#                 btn2 = gr.Button("Chạy sinh ra dữ liệu giả", variant="primary")
#             with gr.Column():
#                 out2 = gr.Image(label="Ảnh giả được tạo")
        
#         btn2.click(run_bash_task_2, inputs=None, outputs=out2)

# if __name__ == "__main__":
#     demo.launch()

BASE_DIR = os.path.abspath(os.getcwd())

if not os.path.exists(os.path.join(BASE_DIR, "Diff-Palm/DiffModels")) and os.path.exists("Diff-Palm.zip"):
    print(">>> Phát hiện file Zip. Đang giải nén Diff-Palm...")
    try:
        with zipfile.ZipFile("Diff-Palm.zip", 'r') as zip_ref:
            zip_ref.extractall(BASE_DIR)
        print(">>> Giải nén thành công!")
    except Exception as e:
        print(f">>> Lỗi giải nén: {e}")

# Kiểm tra lại lần nữa cho chắc
if not os.path.exists(os.path.join(BASE_DIR, "Diff-Palm")):
    print(">>> CẢNH BÁO: Vẫn chưa thấy folder Diff-Palm sau khi giải nén!")
else:
    print(">>> Cấu trúc file OK.")


TEMP_INPUT_DIR = os.path.join(BASE_DIR, "temp_inputs")
TEMP_PCEM_DIR = os.path.join(BASE_DIR, "temp_pcem_out")
TEMP_POLY_DIR = os.path.join(BASE_DIR, "temp_poly_out")
FINAL_OUT_DIR = os.path.join(BASE_DIR, "temp_final_out")

PCEM_SCRIPT = os.path.join(BASE_DIR, "utils", "pcem.py") 
RUN_BASH_SCRIPT = os.path.join(BASE_DIR, "run_diff_palm.sh")


if os.path.exists(RUN_BASH_SCRIPT):
    os.chmod(RUN_BASH_SCRIPT, 0o755)

def clear_folders():
    for folder in [TEMP_INPUT_DIR, TEMP_PCEM_DIR, FINAL_OUT_DIR]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

def zip_results(source_dir, output_filename):
    shutil.make_archive(output_filename, 'zip', source_dir)
    return f"{output_filename}.zip"

def process_augment(input_image, num_images):
    if input_image is None:
        return None, None
    
    num_images = int(num_images)

    clear_folders()

    # 1. Lưu ảnh input
    input_path = os.path.join(TEMP_INPUT_DIR, "input.png")
    Image.fromarray(input_image).save(input_path)
    
    # 2. Chạy PCEM
    print(">>> [Step 1] Đang chạy PCEM...")
    cmd_pcem = [
        sys.executable, PCEM_SCRIPT, 
        "--spath", TEMP_INPUT_DIR, 
        "--dpath", TEMP_PCEM_DIR
    ]
    subprocess.run(cmd_pcem, check=True)

    # Đọc ảnh PCEM
    pcem_img_path = os.path.join(TEMP_PCEM_DIR, "input.png")
    pcem_display = None
    if os.path.exists(pcem_img_path):
        pcem_display = Image.open(pcem_img_path)
    
    # ==> YIELD LẦN 1: Đẩy ảnh PCEM ra màn hình ngay lập tức
    # Gallery và Zip để trống (None) vì chưa có
    print(">>> [UI Update] Hiển thị ảnh PCEM trước...")
    yield pcem_display, [], None
    
    # 3. Chạy Diff-Palm (Augmentation)
    print(f">>> [Step 2] Đang sinh {num_images} ảnh với Diff-Palm...")
    cmd_bash = [
        "bash", RUN_BASH_SCRIPT,
        "1", 
        FINAL_OUT_DIR, 
        TEMP_PCEM_DIR, 
        str(num_images), 
        "1" 
    ]
    subprocess.run(cmd_bash, check=True)
    
    # 4. Hiển thị kết quả
    result_folder = os.path.join(FINAL_OUT_DIR, "results")
    image_paths = glob.glob(os.path.join(result_folder, "*.png"))
    
    display_images = []
    # Hiển thị tất cả ảnh tìm được
    for path in image_paths:
        display_images.append(Image.open(path))
        
    zip_path = zip_results(FINAL_OUT_DIR, "all_results")
    print(">>> [UI Update] Hiển thị kết quả cuối cùng...")
    yield pcem_display, display_images, zip_path


def process_synthetic(num_ids, imgs_per_id):
    num_ids = int(num_ids)
    imgs_per_id = int(imgs_per_id)

    clear_folders()

    print(f">>> [Mode 2] Bắt đầu quy trình sinh {num_ids} ID x {imgs_per_id} ảnh...")
    cmd_bash = [
        "bash", RUN_BASH_SCRIPT,
        "2",
        FINAL_OUT_DIR,
        TEMP_POLY_DIR,  # Bash sẽ sinh nếp nhăn vào đây
        str(num_ids),
        str(imgs_per_id)
    ]
    subprocess.run(cmd_bash, check=True)

    wrinkle_paths = sorted(glob.glob(os.path.join(TEMP_POLY_DIR, "*.png")))
    display_wrinkles = [Image.open(p) for p in wrinkle_paths[:8]] # Lấy tối đa 8 cái để hiển thị

    result_folder = os.path.join(FINAL_OUT_DIR, "results")
    final_paths = sorted(glob.glob(os.path.join(result_folder, "*.png")))

    display_final = []
    if len(final_paths) > 0:
        # Lấy ảnh đại diện theo bước nhảy (imgs_per_id)
        # Ví dụ: sinh 5 ảnh/ID -> lấy ảnh thứ 0, 5, 10...
        representative_paths = final_paths[::imgs_per_id]
        display_final = [Image.open(p) for p in representative_paths[:8]]
    
    zip_path = zip_results(FINAL_OUT_DIR, "synthetic_results")
    
    print(">>> [UI] Hoàn tất Mode 2.")
    return display_wrinkles, display_final, zip_path

with gr.Blocks(title="Diff-Palm System") as demo:
    gr.Markdown("# 🌴 Diff-Palm Data Augmentation System")
    
    with gr.Tabs():
        
        # --- TAB 1: MODE 1 ---
        with gr.TabItem("Mode 1: Augmentation (Từ ảnh thật)"):
            with gr.Row():
                with gr.Column(scale=1):
                    m1_inp = gr.Image(type="numpy", label="Input Image")
                    m1_pcem = gr.Image(label="PCEM Extracted", interactive=False)
                    m1_slider = gr.Slider(1, 16, value=4, step=1, label="Số lượng biến thể (Total Samples)")
                    m1_btn = gr.Button("🚀 Chạy Mode 1", variant="primary")
                
                with gr.Column(scale=2):
                    m1_out = gr.Gallery(label="Output Images", columns=4, height="auto")
                    m1_zip = gr.File(label="Download Result")
            
            m1_btn.click(
                fn=process_augment,
                inputs=[m1_inp, m1_slider],
                outputs=[m1_pcem, m1_out, m1_zip]
            )

        # --- TAB 2: MODE 2 ---
        with gr.TabItem("Mode 2: Synthetic (Tạo giả)"):
            gr.Markdown("Tự động sinh nếp nhăn ngẫu nhiên và tạo ảnh lòng bàn tay tương ứng.")
            with gr.Row():
                with gr.Column(scale=1):
                    m2_ids = gr.Slider(1, 20, value=4, step=1, label="Số lượng ID (Nếp nhăn)")
                    m2_per_id = gr.Slider(1, 10, value=1, step=1, label="Số ảnh trên mỗi ID")
                    m2_btn = gr.Button("🚀 Chạy Mode 2", variant="primary")
                    
                with gr.Column(scale=2):
                    m2_wrinkles = gr.Gallery(label="Mẫu Nếp nhăn (PolyGen)", columns=4, height="auto")
                    m2_final = gr.Gallery(label="Ảnh Giả tương ứng (Representative)", columns=4, height="auto")
                    m2_zip = gr.File(label="Download Full Zip")
            
            m2_btn.click(
                fn=process_synthetic,
                inputs=[m2_ids, m2_per_id],
                outputs=[m2_wrinkles, m2_final, m2_zip]
            )

if __name__ == "__main__":
    demo.launch()

