GeometricEnsemble-XACLE

Setup

1. Clone the repository

git clone [https://github.com/Hmzaah/GeometricEnsemble-XACLE.git](https://github.com/Hmzaah/GeometricEnsemble-XACLE.git)


cd GeometricEnsemble-XACLE


2. Create environment

conda create -n GeoEnsemble python=3.9


conda activate GeoEnsemble


3. Install requirements

pip install -r requirements.txt


4. Install Torch

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)


5. Download Dataset: 

Refer to the official XACLE dataset download procedure from their GitHub repository and add it to GeometricEnsemble-XACLE/data/raw/; please check (Directory_Structure) to understand the structure.

6. Add MS-CLAP model from GitHub

MS-CLAP

git clone [https://github.com/microsoft/CLAP.git](https://github.com/microsoft/CLAP.git)


Please follow the weight download procedure in their repository: MS-CLAP

Other Encoders (Whisper & DeBERTa)
These will be automatically downloaded via HuggingFace transformers when running the feature extraction script.

Usage

Feature Extraction

python src/extract_features.py --config src/config.py


Extracts 9,220-dim features using Whisper, CLAP, and DeBERTa.

Training

python src/train.py


Trains the Heterogeneous Ensemble (XGBoost + SVR).

Inference

python src/fusion.py <input_features> <output_path>


e.g. python src/fusion.py data/features/test_features_9k.npy outputs/inference_result.csv

where <input_features> = data/features/test_features_9k.npy

<output_path> = outputs/inference_result.csv

Evaluation (Taken from XACLE's official implementation)

python src/utils/evaluate.py <inference_csv_path> <ground_truth_csv_path> <save_results_dir>


e.g. python src/utils/evaluate.py outputs/inference_result.csv data/meta_data/validation_average.csv outputs/

where <inference_csv_path> = outputs/inference_result.csv

<ground_truth_csv_path> = data/meta_data/validation_average.csv

<save_results_dir> = outputs/

Results 🥈

<table style="text-align: center;">
<thead>
<tr>
<th rowspan="2">Version</th>
<th colspan="4">Validation</th>
<th colspan="4">Test</th>
</tr>
<tr>
<td>SRCC 

$$\uparrow$$

</td>
<td>LCC 

$$\uparrow$$

</td>
<td>KTAU 

$$\uparrow$$

</td>
<td>MSE 

$$\downarrow$$

</td>
<td>SRCC 

$$\uparrow$$

</td>
<td>LCC 

$$\uparrow$$

</td>
<td>KTAU 

$$\uparrow$$

</td>
<td>MSE 

$$\downarrow$$

</td>
</tr>
</thead>
<tbody>
<tr>
<td>Baseline</td>
<td>0.384</td>
<td>0.396</td>
<td>0.264</td>
<td>4.836</td>
<td>0.334</td>
<td>0.342</td>
<td>0.229</td>
<td>4.811</td>
</tr>
<tr>
<td>GeometricEnsemble (Ours)</td>
<td>

$${\color{blue}0.668}$$

</td>
<td>

$${\color{blue}0.685}$$

</td>
<td>

$${\color{blue}0.490}$$

</td>
<td>3.050</td>
<td>

$${\color{blue}0.653}$$

</td>
<td>0.672</td>
<td>

$${\color{blue}0.468}$$

</td>
<td>3.080</td>
</tr>
</tbody>
</table>

Note:

The results shown above are computed by us for the Validation set, and for the Test set, the results are directly taken from the official leaderboard.

Specifications

Hardware

CPU: Intel(R) Xeon(R) Gold 6154

GPU: Tesla A100-40GB

Time Complexity (Approx)

Type

Time (min)

Feature Ext. ⚡

120

Training 🔥

45

Inference ❄️

10

Directory Structure

GeometricEnsemble-XACLE
    |___src
        |___config.py
        |___extract_features.py
        |___geometry.py
        |___fusion.py
        |___train.py
        |___utils
            |___evaluate.py
    
    |___data
        |___raw
            |___wav
                |___train
                    |___07407.wav
                    |___ . . .
                |___validation
                    |___ . . .
                |___test
                    |___ . . .
            |___meta_data
                |___train.csv
                |___validation_average.csv
        |___features
            |___train_features_9k.npy
            |___test_features_9k.npy

    |___models
        |___xgboost_model.json
        |___svr_model.pkl

    |___notebooks
        |___exploratory_data_analysis.ipynb

    |___assets
        |___architecture_diagram.png
        
    |___requirements.txt
