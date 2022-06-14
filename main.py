from cProfile import run
import imp
import os
import random
import cv2
import glob
from turtle import width
from matplotlib import image
import streamlit as st
import numpy as np
from IPython.display import display, Image
import matplotlib.pyplot as plt
from streamlit_cropper import st_cropper
from st_clickable_images import clickable_images
from pagination import paginator
from PIL import Image
from painting import run_painting
import re

st.title('Learning To Paint - Demo')
options = ('Upload Image', 'MNIST Sample', 'SVHN Sample', 'CelebA Sample')
aspect_dict = aspect_dict = {"1:1": (1, 1), "16:9": (16, 9), "4:3": (4, 3), "2:3": (2, 3), "Free": None}
stroke_types = ('Renderer', 'Round', 'Triangle', 'Bezierwotrans')

choice_img = st.sidebar.selectbox('Select Image', options)

def load_image(image_file, resize_to=None):
    img_display = Image.open(image_file)
    img_model = cv2.imread(image_file, cv2.IMREAD_COLOR)
    if resize_to:
        img_resized = img_display.resize(resize_to)
        return img_display, img_resized, img_model
    return img_display, img_model

def add_control(header, dataset, is_upload=False, val_max_step=1, val_divide=1):
    stroke_types = ('Renderer', 'Round', 'Triangle', 'Bezierwotrans')
    st.subheader(header)

    if is_upload:
        image_file = st.sidebar.file_uploader(
            "Upload Images", type=["png", "jpg", "jpeg"])
        check_crop_image = st.sidebar.checkbox('Crop image')

    choice_model = st.sidebar.selectbox(
        'Model Train On', dataset)
    choice_stroke_type = st.sidebar.selectbox(
        'Stroke Type', stroke_types)

    max_step = st.sidebar.number_input('Step', value=val_max_step)
    divide = st.sidebar.number_input('Divide Image', value=val_divide)

    if is_upload:
        return image_file, check_crop_image, choice_model, choice_stroke_type, max_step, divide
    return choice_model, choice_stroke_type, max_step, divide

def painting_and_show(image_to_paint, choice_model, choice_stroke_type, max_step, divide):
    # st.image(image_to_paint,  width=256)

    path_actor = os.path.join("model", f"{choice_model.upper()}_{choice_stroke_type.lower()}.pkl")
    path_renderer = os.path.join("stroke_types", f"{choice_stroke_type.lower()}.pkl")
    with st.spinner('Wait for painting...'):
        run_painting(path_actor, path_renderer, image_to_paint, max_step, divide)

    # Showing result
    st.subheader("Result")
    col_image_result, col_video_painting = st.columns(2)

    # Showing last canvas
    img_display, _ = load_image(f'output/generated{max_step*5 - 1}.png')
    col_image_result.image(img_display, use_column_width=True)

    # Create and showing video
    create_video_str = f"C:\\ffmpeg\\bin\\ffmpeg.exe -y -r 30 -f image2 -i output/generated%d.png -s 512x512 -c:v libx264 -pix_fmt yuv420p output/video.mp4 -q:v 0 -q:a 0"
    os.system(create_video_str)
    video_bytes = open('output/video.mp4', 'rb').read()
    col_video_painting.video(video_bytes)

    # Showing strokes
    st.subheader('Strokes')
    img_full_path = sorted(glob.glob('output/*.png'), key = lambda x: int(re.findall(r'\d+', x)[0]), reverse=False)
    captions = [idx + 1 for idx in range(len(img_full_path))]

    idx = 0 
    for _ in range(len(img_full_path)-1): 
        cols = st.columns(5)
        for i in range(4):
            if idx < len(img_full_path): 
                cols[i].image(img_full_path[idx], width=128, caption=captions[idx])
                idx += 1
        if idx < len(img_full_path): 
            cols[4].image(img_full_path[idx], width=128, caption=captions[idx])
            idx += 1
        else:
            break


if choice_img == "Upload Image":
    image_file, check_crop_image, \
        choice_model, choice_stroke_type, \
        max_step, divide = add_control(header="Upload Image", dataset=['MNIST', 'SVHN', 'CelebA'],
                                       is_upload=True, val_max_step=8, val_divide=2)

    crop_complete = False
    placeholder = st.empty()
    if image_file is not None:
        cropped_img = None
        if check_crop_image:
            with placeholder.container():
                st.write('Cropping image')
                img = Image.open(image_file)
                width, height = img.size

                magnify = 1
                while True:
                    if width * magnify < 256 or height * magnify < 128:
                        magnify += 1
                    else:
                        break
                
                newsize = (width*magnify, height*magnify)
                img_resized = img.resize(newsize)

                aspect_choice = st.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
                cropped_img = st_cropper(img_resized, box_color="#FFFFFF", aspect_ratio=aspect_dict[aspect_choice])

                st.write("Preview")
                _ = cropped_img.thumbnail((256, 256))
                st.image(cropped_img, width=256)
                crop_complete = st.button('Complete And Painting')

            if crop_complete:
                placeholder.empty()
                st.image(cropped_img,  width=256)
                img_crop_model = cv2.cvtColor(np.array(cropped_img), cv2. COLOR_BGR2RGB)
                painting_and_show(img_crop_model, choice_model, choice_stroke_type, max_step, divide)
        else:
            with placeholder.container():
                image_uploaded = Image.open(image_file)
                st.image(image_uploaded,  width=256)

                path_actor = os.path.join("model", f"{choice_model.upper()}_{choice_stroke_type.lower()}.pkl")
                path_renderer = os.path.join("stroke_types", f"{choice_stroke_type.lower()}.pkl")

                painting = st.button('Painting')
                if painting:
                    img_model = cv2. cvtColor(np.array(image_uploaded), cv2. COLOR_BGR2RGB)
                    painting_and_show(img_model, choice_model, choice_stroke_type, max_step, divide)


elif choice_img == "MNIST Sample":
    choice_model, choice_stroke_type, max_step, divide = add_control(header="MNIST Sample", dataset=['MNIST'],
                                                                     val_max_step=1, val_divide=1)
    
    image_location = st.empty()
    mnist_imgs = os.listdir(os.path.join('sample_images', 'MNIST'))
    path_img = random.choice(mnist_imgs)
    full_path = os.path.join('sample_images', 'MNIST', path_img)
    img_display, img_resize, img_model = load_image(full_path, resize_to=(256, 256))
    image_location.image(img_resize)
    
    btn_contains = st.columns([1, 1, 3])
    painting = btn_contains[0].button("Painting")
    if painting:
        painting_and_show(img_model, choice_model, choice_stroke_type, max_step, divide)
                
elif choice_img == "SVHN Sample":
    choice_model, choice_stroke_type, max_step, divide = add_control(header="SVHN Sample", dataset=['SVHN'],
                                                                     val_max_step=8, val_divide=2)

    image_location = st.empty()
    svhn_imgs = os.listdir(os.path.join('sample_images', 'SVHN'))
    path_img = random.choice(svhn_imgs)
    full_path = os.path.join('sample_images', 'SVHN', path_img)
    img_display, img_resize, img_model = load_image(full_path, resize_to=(256, 256))
    image_location.image(img_resize)
    
    btn_contains = st.columns([1, 1, 3])
    painting = btn_contains[0].button("Painting")
    if painting:
        painting_and_show(img_model, choice_model, choice_stroke_type, max_step, divide)

elif choice_img == "CelebA Sample":
    choice_model, choice_stroke_type, max_step, divide = add_control(header="CelebA Sample", dataset=['CelebA'],
                                                                     val_max_step=40, val_divide=4)

    image_location = st.empty()
    celeba_imgs = os.listdir(os.path.join('sample_images', 'CELEBA'))
    path_img = random.choice(celeba_imgs)
    full_path = os.path.join('sample_images', 'CELEBA', path_img)
    img_display, img_resize, img_model = load_image(full_path, resize_to=(256, 256))
    image_location.image(img_resize)
    
    btn_contains = st.columns([1, 1, 3])
    painting = btn_contains[0].button("Painting")
    if painting:
        painting_and_show(img_model, choice_model, choice_stroke_type, max_step, divide)

