# ocr_model_quantifier
A simple Python project for comparing the output of Keras OCR, Doctr OCR, Easyocr, Tesseract OCR and Kraken OCR.
* platform: win-64

## Installation
1. Clone the repo: 
    ```git clone https://github.com/william-murray1204/ocr_model_quantifier.git```
2. Create an environment using:
    ```conda create --name <env> --file requirements.txt```

3. Install Tesseract from this link https://github.com/UB-Mannheim/tesseract/wiki
4. Add the Tesseract installation directory to your path in environment variables 


## Usage

Add your images you want to compare the outputs of to the "images" folder.

To run the project, navigate to the project directory: ```cd ocr_model_quantifier```

Then run the script:
```python main.py```


* The script will compare the Keras OCR, Doctr OCR, Easyocr, Tesseract OCR and Kraken OCR text predictions for each image in the "images" folder using Matplotlib. 
* The names of the images do not matter as long as they are in the "images" folder.
* It will also track the individual runtime of each model per image in a file named "individual_runtime.csv". The average runtime of each model is written to a file named "average_runtime.txt".

## Example Image Comparisons

![alt text](https://github.com/william-murray1204/ocr_model_comparisons/blob/main/github_example_images/Figure_1.png)

![alt text](https://github.com/william-murray1204/ocr_model_comparisons/blob/main/github_example_images/Figure_2.png)

![alt text](https://github.com/william-murray1204/ocr_model_comparisons/blob/main/github_example_images/Figure_3.png)

![alt text](https://github.com/william-murray1204/ocr_model_comparisons/blob/main/github_example_images/Figure_4.png)

![alt text](https://github.com/william-murray1204/ocr_model_comparisons/blob/main/github_example_images/Figure_5.png)