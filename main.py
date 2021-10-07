from style_transfer import neural_style_transfer, StyleTransferConfig
from tensorflow import keras

config = StyleTransferConfig(
    num_iterations=100,
    save_interval=10,
    start_from_content=True,
    optimizer = keras.optimizers.Adam(
        keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=3.0,
            decay_steps=100,
            decay_rate=0.96  # 0.96
        )
    ),
    # style_weight=0,
    # total_variation_weight=0
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
