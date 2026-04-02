import textile
from textile.utils.image_utils import read_and_process_image

loss_textile = textile.Textile()
image = read_and_process_image("CNN_Images/whole_map.png")
textile_value = loss_textile(image)

print("타일 점수:", textile_value.item())