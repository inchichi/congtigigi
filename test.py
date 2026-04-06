import textile
from textile.utils.image_utils import read_and_process_image

loss_textile = textile.Textile()
image = read_and_process_image("CNN_Images/controlled_eval/Struct_broken/struct_broken.png")
textile_value = loss_textile(image)

print("타일 점수:", textile_value.item())
