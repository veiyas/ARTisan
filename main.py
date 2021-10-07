from style_transfer import neural_style_transfer, StyleTransferConfig

config = StyleTransferConfig(
    optimizer_name='adam',
    num_iterations=20000,
    save_interval=1000
)

# neural_style_transfer(
#     content_img_url="https://www.svtstatic.se/image/wide/992/15663340/1575374592?format=auto",
#     style_img_url="https://i.imgur.com/9ooB60I.jpg",
#     config=config
# )

neural_style_transfer(
    content_img_url="https://upload.wikimedia.org/wikipedia/commons/2/2f/Rijksdag_Stockholm.jpg",
    style_img_url="https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg",
    config=config
)
