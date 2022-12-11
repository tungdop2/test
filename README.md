# INT 3045 - Sentiment Analysis Project Report

## Requirements
- Python 3.8 for stability

## Usage
### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Download checkpoint at [here](https://drive.google.com/file/d/1vPwhogidV-dwaE0n_cykoCRIUiO-ZPgV/view?usp=sharing)
### 3. Inference
For simplicity, just run
```bash
python infer.py
```
Or you can specify 4 arguments:
```bash
python infer.py --df_path <path to csv file> 
                --ckpt_path <path to model checkpoint>
                --output_path <path to output csv file>
                --device <cpu or cuda> 

```
## Kaggle notebook
[INT3045.ipynb](INT3045.ipynb)

## Dataset
[Foody crawled](https://www.kaggle.com/datasets/dtthhh/foody-crawl)
