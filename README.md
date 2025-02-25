# Brain state modeling from intracranial EEG data through high-dimensional embeddings  - BrainHack 2025

## Project Background

Thanks for checking this project out! Our lab collects intracranial EEG data from stereotactic EEG (SEEG) electrodes placed across the brain in patients with drug refractory epilepsy being evaluated in the epilepsy monitoring unit (EMU) prior to potential surgical intervention. These patients can remain in the EMU for up to 2 weeks, providing a rare opportunity to use intracranial EEG data over an extended period of time to learn about not only seizure dynamics, but normal brain function.

However, whole-brain SEEG data proves to be very difficult to analyze due to complex, non-linear relationships in EEG activity between electrodes that evolve across multiple temporal scales. In an effort to interpret this data more effectively, our lab has developed a machine learning model that takes as input 10-second windows of whole-brain SEEG data and summarizes the patient's functional brain state during these 10 seconds as an 'embedding' in 1024-dimensional space. We thus call these 'brain state embeddings'. By feeding 10-second windows across the patient's entire EMU stay, we generate a 1024-dimensional point cloud where each point is a brain state embedding.

1024 dimensions is obviously difficult to work with, so we employ Pairwise Controlled Manifold Approximation (PaCMAP) to project the data to a lower-dimensional manifold that can be more readily analyzed. PaCMAP is a manifold learning technique conceptually similar to UMAP that more effectively preserves both local and global structure during dimensionality reduction.

## Why Sleep Stages

There are many directions for this project, but we are first interested in validating the ability to discriminate between functionally distinct brain states using this point cloud. In other words, are functionally similar brain states closer to each other in the point cloud than functionally dissimilar brain states? Luckily, patients naturally cycle through a series of reliably dissimilar functional brain states: sleep stages!

To explore this, we used a sleep stage classifier on the raw SEEG data to generate metadata about periods of the EMU stay during which the patient was in REM, NREM, and wake. We can then tag the 2-dimensional manifold to visualize whether brain state embeddings from distinct sleep stages appropriately cluster apart from each other. Here is an example:

![alt text](image.png)

## Next Steps

As you can see, brain state embeddings across sleep stages appear to cluster appropriately. However, we are still looking for ways to quantify this analysis - and ultimately learn something about the underlying neural dynamics of sleep. 

1. Sleep stage classifier:
Although we can visualize different clusters of sleep stages based on the metadata, it would be interesting to build a classifier that categorizes brain state embeddings into sleep stages, and then compare these embedding-based categorizations to the metadata. This could be a supervised classifier that uses the metadata as labels, or an unsupervised classifier that is later compared to the metadata. Open questions include how well the classifier can perform and how the performance varies across embedding dimensions (for example, trying to classify embeddings following PaCMAP projection to 2 dimensions versus 10 dimensions).

2. Feature analysis:
The embeddings are defined by dimensions that are highly abstracted from the original SEEG data. However, it would be interesting to investigate whether a given sleep stage is particularly associated with a given set of dimensions. In other words, does our model capture unique features about a sleep stage through a specific set of dimensions, and can we identify these dimensions? We could then track this back to our original raw SEEG data to identify those features, though that would require a more involved project.
We are open to suggestions on how this could be done! One way could be to read out the dimensions that are heavily weighted by a classifier as described above, another interesting approach could be to use recently-published Structure Index (https://github.com/PridaLab/structure_index). Any insights are appreciated!

3. Alternative manifold learning approaches and hyperparameter optimization:
We have been using PaCMAP, but would be open to suggestions about better-suited manifold learning techniques or ways to get the most out of PaCMAP. 

4. Any other cool manifold-based analyses:
Our lab is new to working with manifold data. Insights regarding research directions from those with experience working with manifold data are greatly appreciated!

5. Pipeline development and refinement:
Some of the most helpful contributions would be to make this code more efficient, elaborate, and easy to interact with. Any help on that front would go a long way!


# Getting started

1) Install Python 3.11 and an IDE.
2) Create a conda environment with python 3.11.
- conda create -n [env_name] python=3.11
3) Activate conda environment.
- conda activate [env_name]
4) Use the requirements.txt file to install all necessary packages and dependencies:
- pip3 install -r requirements.txt
5) Download the 1024-dimensional embeddings for 4 different patients as a .pkl file from this link:
- https://drive.google.com/drive/folders/1_HqZW5WNq_69rmBpsqu-V8BuGr3kMQAB?usp=drive_link
6) Make sure this file is in the appropriate directory: source_pickles/1024d_embeddings.pkl.
7) Fork repository to your local repository, then clone this repository locally:
- git clone [link]
8) Create your own branch of this repository:
- git checkout -b [name of your branch]
9) Push any changes you make to your forked repo:
- git push -u origin [name of your branch]


# A Guide to This Repository

## create_manifold.py

This script takes the 1024-dimensional embeddings for a given patient and performs the manifold projection to 2 dimensions using PaCMAP. It will call functions from utils/unpickler.py to achieve this, but you won't ever need to interact with unpickler.py. 

Relevant command-line parameters:
- --patient_id: specify patient to run via an integer (choices are 30, 31, 33, 37).
- --all: provide instead of --patient_id to run all patients.
- --mn_ratio: mid-near pair ratio for PaCMAP. Increasing this value places more emphasis on preserving global structure. Default = 12.0.
- --fp_ratio: far-near pair ratio for PaCMAP. Increasing this value places more emphasis on separating clusters. Default = 1.0.
- --n_neighbors: number of neighbors considered for each point. Default = None, which prompts PaCMAP to automatically determine n_neighbors.
- --lr: learning rate for PaCMAP, controlling how much points are moved with each iteration. Default = 0.01.
- --do_10d: optional argument, if present will perform the usual 2D PaCMAP projection plus an additional 10D PaCMAP projection. This 10D PaCMAP projection is used for unsupervised clustering of the point cloud using HDBSCAN. These clusters are then visualized on the 2D point cloud. The intent here is to give the clustering algorithm access to a higher-dimensional version of the embeddings since information is likely lost at 2 dimensions. If this argument is not provided, no clustering technique will be performed on the 2D point cloud.


Output files:
- embeddings_Epat{ID}.pkl: 1024-dimensional embeddings for a given patient, reformatted from source_pickles file and tagged with some necessary DateTime metadata for later steps.
- manifold_Epat{ID}_MN{mn_ratio}_FP{fp_ratio}_LR{lr}_NN{n_neighbors}.pkl: 2-dimensional embeddings for a given patient along with necessary metadata. Note NN0 denotes that n_neighbors was automatically determined by PaCMAP.
- pointcloud_Epat{ID}_MN{mn_ratio}_FP{fp_ratio}_LR{lr}_NN{n_neighbors}.png: visualization of 2D point cloud. If --do_10d argument was provided, will visualize clusters generated by HDBSCAN on the 10D embeddings.

## sleep_tagger.py

This script takes the 2-dimensional embeddings for a given patient and tags them with corresponding sleep stage metadata.

Relevant command-line parameters:
- --path: relative path to the 2D embeddings file to be tagged.

Output files:
- tagged_manifold_Epat{ID}_MN{mn_ratio}_FP{fp_ratio}_LR{lr}_NN{n_neighbors}.pkl: stores the tagged embeddings for a given patient.
- tagged_pointcloud_Epat{ID}_MN{mn_ratio}_FP{fp_ratio}_LR{lr}_NN{n_neighbors}.png: visualizes the tagged 2D embeddings.


## Typical worflow

The way I usually work with these scripts is to run create_manifold.py with a given set of parameters that I am trying out, for example:

python create_manifold.py --patient_id 30  --mn_ratio 20.0  --fp_ratio 2.0  --lr 0.1

This will create manifold_Epat30_MN20_FP2_LR1_NN0.pkl and pointcloud_Epat30.png files at the directory output/Epat30.

I will then run sleep_tagger.py to tag this manifold with sleep metadata, for example:

python sleep_tagger.py --path output/Epat30/manifold_Epat30_MN20_FP2_LR1_NN0.pkl

