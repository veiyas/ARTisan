from style_transfer import neural_style_transfer, StyleTransferConfig

config = StyleTransferConfig(
    num_iterations=100,
    save_interval=10
)

# neural_style_transfer(
#     content_img_url="https://www.svtstatic.se/image/wide/992/15663340/1575374592?format=auto",
#     style_img_url="https://i.imgur.com/9ooB60I.jpg",
#     config=config
# )

neural_style_transfer(
    content_img_url="https://upload.wikimedia.org/wikipedia/commons/d/d7/Green_Sea_Turtle_grazing_seagrass.jpg",
    style_img_url="https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg",
    config=config
)