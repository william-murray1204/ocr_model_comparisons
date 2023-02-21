import cv2
import pytesseract
from matplotlib import pyplot as plt
from glob import glob
import pandas as pd
import numpy as np
import time
import os
import subprocess
from lxml import etree
import re

os.environ['PYTHONIOENCODING'] = 'utf-8' 



img_fns = glob('images/*')
image_count = len(img_fns)
# image_count = 1
individual_runtime_output_file = 'individual_runtime.csv'
average_runtime_output_file = "average_runtime.txt"


# Check if the individual runtime output file exists and create it if it doesn't
if not os.path.exists(individual_runtime_output_file):
    with open(individual_runtime_output_file, 'w') as f:
        f.write("image_name,model,runtime\n")


# From the two diagonal coordinates given, convert into 4 (a box)
def flesh_out_coordinates(coord_list):
    top_right = [[coord_list[1][0], coord_list[0][1]]]
    bottom_left = [[coord_list[0][0], coord_list[1][1]]]
    modified_list = coord_list[:1] + top_right + coord_list[1:] + bottom_left
    return modified_list

# tesseract_ocr
dfs = []
tesseract_runtime_sum = 0
tesseract_test_count = 0
for img in img_fns[:image_count]:
    config = "--psm 6"
    img_id = img.split('\\')[-1].split('.')[0] 
    img_df = pd.DataFrame(columns=['text', 'bbox', 'img_id'])


    # Start tracking runtime and run the model
    start_time = time.time()
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    results = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=config)
    end_time = time.time()
    runtime = end_time - start_time
    tesseract_runtime_sum += runtime

    # Create a csv row for runtime
    data = str(f'"{os.path.basename(img)}","tesseract_ocr",{runtime}\n')
    # Save the csv row to the file
    with open(individual_runtime_output_file, 'a') as f:
        f.write(data)
    
    for i in range(len(results["text"])):
        word = results["text"][i].strip()
        if word:
            x1 = results["left"][i]
            y1 = results["top"][i]
            x2 = x1 + results["width"][i]
            y2 = y1 + results["height"][i]
            # From the two diagonal coordinates given, convert into 4 (a box)
            box = flesh_out_coordinates([[x1, y1], [x2, y2]])
            img_df = pd.concat([
                    img_df, pd.DataFrame({
                        'text': (word.encode('ascii', 'ignore')).decode('utf-8'),
                        'bbox': [box],
                        'img_id': [img_id]
                    })
                ], ignore_index=True)

    dfs.append(img_df)
    tesseract_test_count += 1

tesseract_df = pd.concat(dfs) 


# kraken_ocr
dfs = []
kraken_runtime_sum = 0
kraken_test_count = 0
for img in img_fns[:image_count]:
    img_id = img.split('\\')[-1].split('.')[0] 
    img_df = pd.DataFrame(columns=['text', 'bbox', 'img_id'])

    # Start tracking runtime and run the model
    start_time = time.time()
    command = f'kraken -i "{img}" "image.xml" --hocr binarize segment ocr -m en_best.mlmodel'  # Run kraken OCR model and output hocr results to xml file
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    print(output.decode())  
    end_time = time.time()
    runtime = end_time - start_time
    kraken_runtime_sum += runtime

     # Create a csv row for runtime
    data = str(f'"{os.path.basename(img)}","kraken_ocr",{runtime}\n')
    # Save the csv row to the file
    with open(individual_runtime_output_file, 'a') as f:
        f.write(data)

    # Parse the XML using lxml
    root = etree.parse('image.xml')
    # Use XPath to select all the ocr_line elements
    lines = root.xpath('//span[@class="ocr_line"]')
    # Loop over the lines and extract the words and bounding boxes
    for line in lines:
        words = []
        boxes = []

        # Use XPath to select all the ocrx_word elements within the line
        word_elements = line.xpath('.//span[@class="ocrx_word"]')

        # Loop over the word elements and extract the text and bbox attributes
        for word_element in word_elements:
            text = word_element.text
            bbox_str = word_element.get('title')

            # Remove the x_confs data from the bbox_str
            bbox_parts = bbox_str.split(';')
            bbox_parts = [part for part in bbox_parts if 'x_confs' not in part]
            bbox_str = ';'.join(bbox_parts)

            # Parse the bbox attribute
            bbox = [int(x) for x in re.findall(r'\d+', bbox_str)]

            words.append(text)
            boxes.append(bbox)

        # Add the line to the data object
        for i in range(len(words)):
            if words[i] == " ":
                pass
            else: 
                # From the two diagonal coordinates given, convert into 4 (a box)
                box = flesh_out_coordinates([[boxes[i][0], boxes[i][1]], [boxes[i][2], boxes[i][3]]])
                img_df = pd.concat([
                                    img_df, pd.DataFrame({
                                        'text': words[i],
                                        'bbox': [box],
                                        'img_id': [img_id]
                                    })
                                ], ignore_index=True)

        dfs.append(img_df)
    kraken_test_count += 1
    os.remove("image.xml")

kraken_df = pd.concat(dfs) 


# # convert dataframe to a string
# df_str = kraken_df.to_string(index=False)

# # write the string to a text file
# with open('kraken.txt', 'w') as f:
#     f.write(df_str)




# easyocr
import easyocr
reader = easyocr.Reader(['en'], gpu = True)

dfs = []
easyocr_runtime_sum = 0
easyocr_test_count = 0
for img in img_fns[:image_count]:
    img_id = img.split('\\')[-1].split('.')[0]

    # Start tracking runtime and run the model
    start_time = time.time()
    result = reader.readtext(img)
    end_time = time.time()
    runtime = end_time - start_time
    easyocr_runtime_sum += runtime

    # Create a csv row for runtime
    data = str(f'"{os.path.basename(img)}","easyocr",{runtime}\n')
    # Save the csv row to the file
    with open(individual_runtime_output_file, 'a') as f:
        f.write(data)
    
    img_df = pd.DataFrame(result, columns=['bbox','text','conf'])
    img_df['img_id'] = img_id
    dfs.append(img_df)
    easyocr_test_count += 1
easyocr_df = pd.concat(dfs)


# keras_ocr
import keras_ocr
pipeline = keras_ocr.pipeline.Pipeline()

dfs = []
keras_runtime_sum = 0
keras_test_count = 0
for img in img_fns[:image_count]:

    # Start tracking runtime and run the model
    start_time = time.time()
    results = pipeline.recognize([img])
    end_time = time.time()
    runtime = end_time - start_time
    keras_runtime_sum += runtime
    
    # Create a csv row for runtime
    data = str(f'"{os.path.basename(img)}","keras_ocr",{runtime}\n')
    # Save the csv row to the file
    with open(individual_runtime_output_file, 'a') as f:
        f.write(data)

    result = results[0]
    img_id = img.split('\\')[-1].split('.')[0]
    img_df = pd.DataFrame(result, columns=['text', 'bbox'])
    img_df['img_id'] = img_id
    dfs.append(img_df)
    keras_test_count += 1
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
doctr_runtime_sum = 0
doctr_test_count = 0
for img in img_fns[:image_count]:

    # Start tracking runtime and run the model
    start_time = time.time()
    results_array = DocumentFile.from_images(img)
    results = model(results_array)

    end_time = time.time()
    runtime = end_time - start_time
    doctr_runtime_sum += runtime

    # Create a csv row for runtime
    data = str(f'"{os.path.basename(img)}","doctr_ocr",{runtime}\n')
    # Save the csv row to the file
    with open(individual_runtime_output_file, 'a') as f:
        f.write(data)

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
    doctr_test_count += 1
doctrocr_df = pd.concat(dfs)



# import json
# # convert list to json string
# df_json = doctrocr_df.to_json(orient='records')
# json_object = json.loads(df_json)

# # write json string to json file
# with open('output.json', 'w') as file:
#     json.dump(json_object, file, indent=4)





import matplotlib.patches as patches

# Plot Results: easyocr vs keras_ocr
def plot_compare(img_fn, easyocr_df, kerasocr_df, doctrocr_df, tesseract_df, kraken_ocr):
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

    kraken_results = kraken_df.query('img_id == @img_id')[['text','bbox']].values.tolist()
    kraken_results = [(x[0], np.array(x[1])) for x in kraken_results]
    axs[1,1].imshow(img)
    axs[1,1].set_title('kraken_ocr results', fontsize=24, pad=15)
    axs[1,1].set_xticks([])
    axs[1,1].set_yticks([])
    axs[1,1].set_frame_on(True)
    for result in kraken_results:
        text = result[0]
        bbox = result[1]
        rect = patches.Rectangle((bbox[0][0], bbox[0][1]), bbox[2][0] - bbox[0][0], bbox[2][1] - bbox[0][1], linewidth=1, edgecolor='r', facecolor='none')
        axs[1, 1].add_patch(rect)
        axs[1, 1].text(bbox[0][0], bbox[1][1], text, fontsize=12, color='r', va='bottom', ha='left', bbox=dict(facecolor='white', edgecolor='none', pad=0.5), transform=axs[1,1].transData, clip_on=True)

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



    # Add padding using tight_layout
    plt.tight_layout()

    plt.show()


# Average the runtime for the ocr tests
doctr_ocr_runtime_average = doctr_runtime_sum / doctr_test_count
keras_ocr_runtime_average = keras_runtime_sum / keras_test_count
easyocr_ocr_runtime_average = easyocr_runtime_sum / easyocr_test_count
tesseract_ocr_runtime_average = tesseract_runtime_sum / tesseract_test_count
kraken_ocr_runtime_average = kraken_runtime_sum / kraken_test_count

if image_count < 2:
    append = ''
else:
    append = 's'

accuracy_output = f"Running models against {image_count} image{append} found that:\nAverage Doctr_OCR runtime: {doctr_ocr_runtime_average} seconds\nAverage Easyocr_OCR runtime: {easyocr_ocr_runtime_average} seconds\nAverage keras_OCR runtime: {keras_ocr_runtime_average} seconds\nAverage Tesseract_OCR runtime: {tesseract_ocr_runtime_average} seconds\nAverage Kraken_OCR runtime: {kraken_ocr_runtime_average} seconds"

# Check if the output file exists and create it if it doesn't
if not os.path.exists(average_runtime_output_file):
    with open(average_runtime_output_file, 'w') as f:
        f.write(accuracy_output)
else: 
    with open(average_runtime_output_file, 'a') as f:
        f.write("\n-----------------------------------------------------------------------------\n")
        f.write(accuracy_output)

# Loop over results
for img_fn in img_fns[:image_count]:
    plot_compare(img_fn, easyocr_df, kerasocr_df, doctrocr_df, tesseract_df, kraken_df)

