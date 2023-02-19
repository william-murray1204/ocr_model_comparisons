import cv2
import pytesseract
from matplotlib import pyplot as plt
from glob import glob
import pandas as pd
import numpy as np


image_count = 5

img_fns = glob('images/*')


# Convert relative bbox coordinates to pixel coordinates
def flesh_out_coordinates(coord_list):
    top_right = [[coord_list[1][0], coord_list[0][1]]]
    bottom_left = [[coord_list[0][0], coord_list[1][1]]]
    modified_list = coord_list[:1] + top_right + coord_list[1:] + bottom_left
    return modified_list

# tesseract_ocr
dfs = []
for img in img_fns[:image_count]:
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    config = "--psm 6"
    img_id = img.split('\\')[-1].split('.')[0] 
    img_df = pd.DataFrame(columns=['text', 'bbox', 'img_id'])
    
    results = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=config)

    for i in range(len(results["text"])):
        word = results["text"][i].strip()
        if word:
            x1 = results["left"][i]
            y1 = results["top"][i]
            x2 = x1 + results["width"][i]
            y2 = y1 + results["height"][i]
            # Swap x and y coordinates, and invert y
            box = flesh_out_coordinates([[x1, y1], [x2, y2]])
            img_df = pd.concat([
                    img_df, pd.DataFrame({
                        'text': (word.encode('ascii', 'ignore')).decode('utf-8'),
                        'bbox': [box],
                        'img_id': [img_id]
                    })
                ], ignore_index=True)

    dfs.append(img_df)
tesseract_df = pd.concat(dfs) 



# easyocr
import easyocr
reader = easyocr.Reader(['en'], gpu = False)

dfs = []
for img in img_fns[:image_count]:
    result = reader.readtext(img)
    img_id = img.split('\\')[-1].split('.')[0]
    img_df = pd.DataFrame(result, columns=['bbox','text','conf'])
    img_df['img_id'] = img_id
    dfs.append(img_df)
easyocr_df = pd.concat(dfs)


# keras_ocr
import keras_ocr
pipeline = keras_ocr.pipeline.Pipeline()

dfs = []
for img in img_fns[:image_count]:
    results = pipeline.recognize([img])
    result = results[0]
    img_id = img.split('\\')[-1].split('.')[0]
    img_df = pd.DataFrame(result, columns=['text', 'bbox'])
    img_df['img_id'] = img_id
    dfs.append(img_df)
kerasocr_df = pd.concat(dfs)

# doctr_ocr
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

# Convert relative bbox coordinates to pixel coordinates
def convert_coordinates(coord_list, img_width, img_height):
    pixel_coords = []
    for coord in coord_list:
        x1 = float(coord[0] * img_width)
        y1 = float(coord[1] * img_height)

        pixel_coords.append([x1, y1])

    top_right = [[pixel_coords[1][0], pixel_coords[0][1]]]
    bottom_left = [[pixel_coords[0][0], pixel_coords[1][1]]]
    modified_list = pixel_coords[:1] + top_right + pixel_coords[1:] + bottom_left
    return modified_list

dfs = []
for img in img_fns[:image_count]:
    results_array = DocumentFile.from_images(img)
    results = model(results_array)
    result_json = results.export()
    img_df = pd.DataFrame(columns=['text', 'bbox', 'img_id'])
    img_id = img.split('\\')[-1].split('.')[0] 
    dimensions = result_json['pages'][0]['dimensions']
    img_width = dimensions[1]
    img_height = dimensions[0]
    for page in result_json['pages']:
        for block in page['blocks']:
            for line in block['lines']:
                for word in line['words']:
                    box = convert_coordinates((word['geometry']), img_width, img_height)
                    img_df = pd.concat([
                    img_df, pd.DataFrame({
                        'text': [word['value']],
                        'bbox': [box],
                        'img_id': [img_id]
                    })
                ], ignore_index=True)
    dfs.append(img_df)
doctrocr_df = pd.concat(dfs)



import matplotlib.patches as patches

# Plot Results: easyocr vs keras_ocr
def plot_compare(img_fn, easyocr_df, kerasocr_df, doctrocr_df, tesseract_df):
    fig, axs = plt.subplots(2, 3)
    img_id = img_fn.split('\\')[-1].split('.')[0] 

    easy_results = easyocr_df.query(f'img_id == @img_id')[['text','bbox']].values.tolist()
    easy_results = [(x[0], np.array(x[1])) for x in easy_results]
    img = plt.imread(img_fn)
    axs[0,0].imshow(img)
    axs[0,0].set_title('easyocr results', fontsize=24, pad=15)
    axs[0,0].set_xticks([])
    axs[0,0].set_yticks([])
    axs[0,0].set_frame_on(True)
    for result in easy_results:
        text = result[0]
        bbox = result[1]
        rect = patches.Rectangle((bbox[0][0], bbox[0][1]), bbox[2][0] - bbox[0][0], bbox[2][1] - bbox[0][1], linewidth=1, edgecolor='r', facecolor='none')
        axs[0, 0].add_patch(rect)
        axs[0, 0].text(bbox[0][0], bbox[1][1], text, fontsize=12, color='r', va='bottom', ha='left', bbox=dict(facecolor='white', edgecolor='none', pad=0.5), transform=axs[0,0].transData, clip_on=True)

    keras_results = kerasocr_df.query('img_id == @img_id')[['text','bbox']].values.tolist()
    keras_results = [(x[0], np.array(x[1])) for x in keras_results]
    axs[0,2].imshow(img)
    axs[0,2].set_title('keras_ocr results', fontsize=24, pad=15)
    axs[0,2].set_xticks([])
    axs[0,2].set_yticks([])
    axs[0,2].set_frame_on(True)
    for result in keras_results:
        text = result[0]
        bbox = result[1]
        rect = patches.Rectangle((bbox[0][0], bbox[0][1]), bbox[2][0] - bbox[0][0], bbox[2][1] - bbox[0][1], linewidth=1, edgecolor='r', facecolor='none')
        axs[0, 2].add_patch(rect)
        axs[0, 2].text(bbox[0][0], bbox[1][1], text, fontsize=12, color='r', va='bottom', ha='left', bbox=dict(facecolor='white', edgecolor='none', pad=0.5), transform=axs[0,2].transData, clip_on=True)

    doctr_results = doctrocr_df.query('img_id == @img_id')[['text','bbox']].values.tolist()
    doctr_results = [(x[0], np.array(x[1])) for x in doctr_results]
    axs[1,0].imshow(img)
    axs[1,0].set_title('doctr_ocr results', fontsize=24, pad=15)
    axs[1,0].set_xticks([])
    axs[1,0].set_yticks([])
    axs[1,0].set_frame_on(True)
    for result in doctr_results:
        text = result[0]
        bbox = result[1]
        rect = patches.Rectangle((bbox[0][0], bbox[0][1]), bbox[2][0] - bbox[0][0], bbox[2][1] - bbox[0][1], linewidth=1, edgecolor='r', facecolor='none')
        axs[1, 0].add_patch(rect)
        axs[1, 0].text(bbox[0][0], bbox[1][1], text, fontsize=12, color='r', va='bottom', ha='left', bbox=dict(facecolor='white', edgecolor='none', pad=0.5), transform=axs[1,0].transData, clip_on=True)

    tesseract_results = tesseract_df.query('img_id == @img_id')[['text','bbox']].values.tolist()
    tesseract_results = [(x[0], np.array(x[1])) for x in tesseract_results]
    axs[1,2].imshow(img)
    axs[1,2].set_title('tesseract_ocr results', fontsize=24, pad=15)
    axs[1,2].set_xticks([])
    axs[1,2].set_yticks([])
    axs[1,2].set_frame_on(True)
    for result in tesseract_results:
        text = result[0]
        bbox = result[1]
        rect = patches.Rectangle((bbox[0][0], bbox[0][1]), bbox[2][0] - bbox[0][0], bbox[2][1] - bbox[0][1], linewidth=1, edgecolor='r', facecolor='none')
        axs[1, 2].add_patch(rect)
        axs[1, 2].text(bbox[0][0], bbox[1][1], text, fontsize=12, color='r', va='bottom', ha='left', bbox=dict(facecolor='white', edgecolor='none', pad=0.5), transform=axs[1,2].transData, clip_on=True)

    axs[0,1].imshow(img)
    axs[0,1].set_title('original image', fontsize=24, pad=15)
    axs[0,1].set_xticks([])
    axs[0,1].set_yticks([])
    axs[0,1].set_frame_on(True)

    axs[1,1].axis('off')

    plt.show()

# Loop over results
for img_fn in img_fns[:image_count]:
    plot_compare(img_fn, easyocr_df, kerasocr_df, doctrocr_df, tesseract_df)


