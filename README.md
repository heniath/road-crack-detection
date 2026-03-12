# Road Crack Detection And Segmentation 
## 1. Requirements
- Python version: 3.10
## 2. Data Preparation
- Download the SUT-Crack dataset at: https://data.mendeley.com/datasets/gsbmknrhkv/6
and place it according to this:  
```
    road-crack-detection
    |
    |-- dataset
    |   |
    |   |-- SUT-Crack.zip
    |   |
    |   |-- SUT-Crack  
```
- In PowerShell, type:  
    `Expand-Archive -Path "./data/SUT-Crack.zip" -DestinationPath "./data/SUT-Crack"`