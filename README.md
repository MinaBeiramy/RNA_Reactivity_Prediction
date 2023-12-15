# RNA_Reactivity_Prediction

### Step 1: (OPTIONAL)data
** This step is only necessary if you want to preprocess the data yourself. Otherwise, go to next step. **
you can access data via this<a>https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/data</a> kaggle competition and place it in `DATA` directory.

### Step 2: Preprocessed Data
Download the data form Kaggle and place it in your `path/to/RNA_reactivity_predition`



### Step 3: Create Conda Env and Install Requirements
```
cd path/to/project
conda create -n rna-reactivity-prediction
conda activate rna-reactivity-prediction
pip install -r requirements.txt
```

### Step 4: utils.py
Uncomment the functions in utils.py and run the script.

### Step 4: Change the variable 
in `train_config.json` , `test_config.json`
** TO BE IMPLEMENTED, FOR NOW GO WITH THE DEFULT SETTING, OR CHANGE MANUALLY**

### Step 5: Run the script
```
python script.py -m {model_name:str} -e {epoch_number:int} -v {mode_vesion_ number}
```

### Eterna installation
Install mini-forge mamba.
Install eterna and other dependencies. 
```
pip install -q --upgrade arnie 
pip install -q --upgrade forgi
conda install -y -q -c bioconda viennarna
pip install -q git+https://github.com/DasLab/draw_rna.git

```
set the following env paths.
```
ETERNAFOLD_PATH=~/miniforge3/envs/eter/bin/eternafold-bin
ETERNAFOLD_PARAMETERS=~/miniforge3/envs/eter/lib/eternafold-lib/parameters/EternaFoldParams.v1
```
