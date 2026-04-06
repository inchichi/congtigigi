import textile
from textile.utils.image_utils import read_and_process_image

# Game tiles keep more object detail when the tiled preview is resized less aggressively.
GAME_TILE_EVAL_CONFIG = {
    "lambda_value": 0.25,
    "resolution": (512, 512),
    "number_tiles": 2,
}

loss_textile = textile.Textile(**GAME_TILE_EVAL_CONFIG)
image = read_and_process_image("CNN_Images/gametile/Bad/rpgTile000_stylized_winter2Dgame.jpg")
textile_value = loss_textile(image)
textile_logit = loss_textile(image, return_logits=True)

print("평가 설정:", GAME_TILE_EVAL_CONFIG)
print("raw logit:", textile_logit.item())
print("타일 점수:", textile_value.item())
