# Chalenge
## Input
- Dataset of ~50,000 audio files with tagging of:
  - audio file
  - speaker
  - language
  - noise (comm/backgroud/clean)
- Validation set of 1,000 records
  - anchor audio file
  - anchor noise (comm/backgroud/clean)
  - group of audio files from the laguage of the anchor
 
## output
- for each validation record find which file (threre is preciceliy one) that is from the speaker as the anchor


## Solution proposal:

1. [optional] clean the input audio files
   - regardless of the noise
   - depends on the noise
2. [optional] sperate the files to different languages
3. [optional] add external files for each language from external resources:
   - [OpenSLR](https://openslr.org/resources.php)
5. Fine tune a nemo model (titanet-small/large/ecapa) using the notebook with the data
   - [Speaker_Recogniton_Verification.ipynb](https://colab.research.google.com/github/NVIDIA/NeMo/blob/main/tutorials/speaker_tasks/Speaker_Identification_Verification.ipynb)
   - [Nvidia Nemo models](https://catalog.ngc.nvidia.com/models?filters=&orderBy=scoreDESC&query=speaker%20recognition)
     - [titanet_small](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_small)
     - [titanet_large](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_large)
     - [ecapa_tdnn](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/ecapa_tdnn)
6. TBD
