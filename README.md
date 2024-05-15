# :video_game: :joystick: AFKAdventures: Steam Game Recommendation System

## Model Approaches (User Playtime based Collobrative Filtering using BPR + Content Filtering using LDA + Clustering using Cosine-Similarity)

### What is Model Approaches?
Our Steam games recommendation system implements collaborative filtering, content-based filtering, and clustering techniques to test their efficacy in providing accurate and personalized game suggestions. Collaborative filtering analyzes user playtime and interactions to recommend games enjoyed by similar users. Content-based filtering suggests games with similar attributes to those a user has previously enjoyed. Clustering techniques group users and games based on behaviors and attributes to uncover hidden trends and enhance recommendations. By evaluating these methods separately, we aim to understand their strengths and limitations before integrating them into a future hybrid model for even better recommendations.

### Data for Steam:
Data is sourced using the Steam Web API and was processed and can be found [here](https://drive.google.com/drive/folders/1PhvTPd60Jr2QaJwgtjsm-sj-WRHADyLO).

### Data Preprocessing 
Data was prepared by merging different datasets gathered and data preprocessing involved removing nulls, duplicates and then selecting relevant features and utilizing them for the analysis. 

## Steps to Dowload the file:

### Installation
Install the Python Environment and download the zip folder from the repository. Using the command prompt, navigate to the directory of the folder.

To download data, run the following command: **`python setup.py extract`**.

### For EDA
To get visualizations and information about the data, run the following command: **`python sources/eda.py`**.

### Create a Virtual Environment to insall dependencies

#### For Mac, run the following command 

1. python3 -m venv myenv
2. source myenv/bin/activate

#### For windows, run the following command
1. python -m venv myenv
2. myenv\Scripts\activate

    
### To install dependencies, run the following:
pip install -r requirements.txt

### For recommmendation using Colloborative filtering, run the following command

**`python models/colloborativefiltering.py`**.