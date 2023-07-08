from PIL import Image, ImageEnhance, ImageOps
import random


# 定义数据增强函数
def augment_data(image):
    # 随机旋转
    angle = random.randint(-10, 10)
    image = image.rotate(angle)

    # 随机水平翻转
    if random.random() < 0.5:
        image = ImageOps.flip(image)

    # 随机垂直翻转
    if random.random() < 0.5:
        image = ImageOps.mirror(image)

    # 随机裁剪
    w, h = image.size
    left = random.randint(0, int(w * 0.2))
    top = random.randint(0, int(h * 0.2))
    right = random.randint(int(w * 0.8), w)
    bottom = random.randint(int(h * 0.8), h)
    image = image.crop((left, top, right, bottom))

    # 随机调整亮度、对比度、饱和度
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))

    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))

    return image


if __name__ == '__main__':
    # 示例代码
    image_path = "example.jpg"
    image = Image.open(image_path)

    # 展示原图
    image.show()

    # 进行数据增强
    augmented_image = augment_data(image)

    # 展示增强后的图像
    augmented_image.show()
