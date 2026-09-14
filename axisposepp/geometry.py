from PIL import Image


def open_image(path):
    image = Image.open(path)
    if image.mode == "RGBA":
        rgb = Image.new("RGB", image.size, (0, 0, 0))
        rgb.paste(image, mask=image.getchannel("A"))
        return rgb
    return image.convert("RGB")
