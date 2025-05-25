import os
from PIL import Image

# 设置保存图像的文件夹路径
image_folder = './outputs/images'

# 获取文件夹中所有的 PNG 图片文件
image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

# 排序图片文件，确保它们按照顺序加载（如果有命名规则，如 cd_0.png, cd_1.png, ...）
image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # 假设文件名格式是 cd_0.png, cd_1.png, ...

images = []
for file in image_files:
    # 逐个打开图像
    img_path = os.path.join(image_folder, file)
    with Image.open(img_path) as img:
        images.append(img.copy())

# 设置输出 GIF 文件的路径
gif_output_path = './cd_animation.gif'

# 将图片列表保存为 GIF
images[0].save(
    gif_output_path,
    save_all=True,
    append_images=images[1:],  # 追加剩余的图片
    duration=200,  # 每帧的持续时间，单位是毫秒
    loop=0  # 设置动画循环次数，0 表示无限循环
)

print(f"GIF 已保存到 {gif_output_path}")
