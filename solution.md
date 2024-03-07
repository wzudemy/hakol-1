# Challenge
## Input
- Dataset of approximately 50,000 audio files tagged with:
  - Audio file
  - Speaker
  - Language
  - Noise (Communication/Background/Clean)
- Validation set of 1,000 records:
  - Anchor audio file
  - Anchor noise (Communication/Background/Clean)
  - Group of audio files from the language of the anchor
  - Gropu noise

## Output
- For each validation record, find the file (there is precisely one) that matches the speaker as the anchor.

## Solution Proposal:
1. [Optional] Clean the input audio files:
   - Irrespective of the noise
   - Based on the noise
2. [Optional] Separate the files into different languages
3. [Optional] Separate the files into different noise
4. [Optional] Add external files for each language from external resources:
   - [OpenSLR](https://openslr.org/resources.php)
   - [commonvoice](https://commonvoice.mozilla.org/en/datasets)
5. Fine-tune a NeMo model (TitanNet-Small/Large/ECAPA) using the notebook with the data (for each langague ? )
   - [Speaker_Recognition_Verification.ipynb](https://colab.research.google.com/github/NVIDIA/NeMo/blob/main/tutorials/speaker_tasks/Speaker_Identification_Verification.ipynb)
   - [NVIDIA NeMo models](https://catalog.ngc.nvidia.com/models?filters=&orderBy=scoreDESC&query=speaker%20recognition)
     - [TitanNet-Small](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_small)
     - [TitanNet-Large](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_large)
     - [ECAPA-TDNN](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/ecapa_tdnn)
6. Create a ML (same xgboost, e.g. catboost) with the following inputs:
   - Speaker ID (calculated by the finetuned model)
   - Other features:
     - [Optional] Gender (calculated by an external model)
     - Other (e.g. Spectral specification, Age)
8. Train the Model on the input dataset
   - Using train/valid split
   - Hyper parameters
10. Infernce:
   - [Optional] Detect the Language
   - [Optinoal] choose the approtiate noise model
   - Perform the model fit
