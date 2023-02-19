# ocr_model_comparisons
A simple Python project for comparing the output of keras_ocr, doctr_ocr, easyocr and tesseract_ocr.
* platform: win-64

## Installation
1. Clone the repo: 
    ```git clone https://github.com/william-murray1204/ocr_model_comparisons.git```
2. Create an environment using:
    ```conda create --name <env> --file requirements.txt```

3. Install Tesseract from this link https://github.com/UB-Mannheim/tesseract/wiki
4. Add the Tesseract installation directory to your path in environment variables 


## Usage
To run the project, navigate to the project directory: ```cd ocr_model_comparisons```

Then run the script:
```python main.py```


* The script will compare the Keras_ocr, Doctr_ocr, Easyocr and Tesseract_ocr text predictions for each image in the "images" folder using Matplotlib. 
* The names of the images do not matter as long as they are in the "images" folder.
* It will also track the individual runtime of each model per image in a file named "individual_runtime.csv". The average runtime of each model is written to a file named "average_runtime.txt".

## Example Image Comparisons

![alt text](https://github.com/william-murray1204/ocr_model_comparisons/blob/main/github_example_images/example_1.PNG)

![alt text](https://github.com/william-murray1204/ocr_model_comparisons/blob/main/github_example_images/example_2.PNG)

![alt text](https://github.com/william-murray1204/ocr_model_comparisons/blob/main/github_example_images/example_3.PNG)

![alt text](https://github.com/william-murray1204/ocr_model_comparisons/blob/main/github_example_images/example_4.PNG)

![alt text](https://github.com/william-murray1204/ocr_model_comparisons/blob/main/github_example_images/example_5.PNG)
