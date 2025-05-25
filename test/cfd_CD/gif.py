import imageio
import os

vtk_folder = "results"
gif_folder = "gif"
images = []
steps = 50

for step in range(steps):
    filename = f"{vtk_folder}/frame_{step:04d}.png"
    if os.path.exists(filename):
        images.append(imageio.imread(filename))

gif_path = os.path.join(gif_folder, "animation.gif")
imageio.mimsave(gif_path, images, fps=10)
print(f"✅ GIF 已生成：{gif_path}")
