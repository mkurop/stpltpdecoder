# stpltpdecoder
Classes for performing analysis and synthesis based on the Short Term Predictor/Long Term Predictor (STP/LTP) speech signal model. This speech signal model orginates from the speech coding paradigm known as the Code Excited Linear Prediction (CELP).

# Runnig the decoder.py
First create conda environment using the command:

```
conda create --name <env_name> --file requirements.txt
```

Activate the environment using

```
conda activate <env_name>
```
Then, go to src folder in the root of the repo and run

```
python decoder.py
```

This will create output sound files in the directories

```
<repo_root>/data/output/female
<repo_root>/data/output/male
```

you can listen to the input files from

```
<repo_root>/data/input/female
<repo_root>/data/input/male
```

and confirm that the output sounds the same.

