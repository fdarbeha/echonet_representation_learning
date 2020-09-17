The datasets used in this paper can be downloaded from the following links:

Echonet:
https://echonet.github.io/dynamic/
After filling the reuquest form, it will take 1-2 days to get access.
Once you have access download the dataset and modify the following variable in data/echonet_dataset.py to point to the location of data:
DATA_DIR = 'PATH/TO/DATA/FOLDER'

OASIS3:
https://www.oasis-brains.org
After applying for access, it will take 2-5 days to get approved.
Once you have access, download the data nd use the clinica open-source tool for preprocessing:
- Clinica t1-volume pipeline: http://www.clinica.run/doc/Pipelines/T1_Volume/

This tool performs bias field correction and nonlinear registration into MNI space.

Download our pretrianed model best_pastssl_3d_200.pth from https://drive.google.com/file/d/1lGtVXoqCCiSUTtZtGp0WUn3ZKKSw6rAk/view?usp=sharing
and place it under output/PaSTSSL_r3d_18_pretrained_run1/

To fine-tune run the eval.sh script

To train your sself-supervised model run train.sh


