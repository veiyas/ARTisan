# ARTisan

Uses deep learning to create art.

## How to use

##### Logistics

Install Python (tested on version 3.7-3.9). Make sure to add python to PATH, it makes everything easier.

Install packages tensorflow and pillow using pip in terminal/cmd/etc:

```shell
pip install tensorflow
pip install pillow
```

If pythons complains that you are missing any other packages you should install them too. Tensorflow might take a considerable time to install.

##### Choose style and content image

Open the file ``` main.py``` and change the arguments ```content_img_url``` and ```style_img_url``` to your desired, direct linked image URL. The images should be any of the common types like jpg, png, etc.

- ```content_img_url``` is the image you want to apply a style on
- ```style_img_url``` is the image you want to extract the style from

It should end up looking something like

```python
neural_style_transfer(
    content_img_url="https://upload.wikimedia.org/wikipedia/commons/2/2f/Rijksdag_Stockholm.jpg",
    style_img_url="https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg",
    config=config
)
```

Save the file and run the script with one of

```shell
python main.py
python3 main.py
```

depending on how you installed python.

##### Warning

Unless you have a powerful GPU this script will take a long time to finish. Expect anything between 1-12 hours depending on your hardware.

