# RNA_Reactivity_Prediction

### Step 1: (OPTIONAL)data
** This step is only necessary if you want to preprocess the data yourself. Otherwise, go to next step. **
you can access data via this<a>https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/data</a> kaggle competition and place it in `DATA` directory.

### Step 2: Preprocessed Data
Download the data form <a>Drive</a> and place it in your `path/to/RNA_reactivity_predition`

### Step 3: Create Conda Env and Install Requirements
```
cd path/to/project
conda create -n rna-reactivity-prediction
conda activate rna-reactivity-prediction
pip install -r requirements.txt
```
### Step 4: Change the variable 
in `train_config.json` , `test_config.json`
** TO BE IMPLEMENTED, FOR NOW GO WITH THE DEFULT SETTING, OR CHANGE MANUALLY**

### Step 5: Run the script
```
python script.py
```
