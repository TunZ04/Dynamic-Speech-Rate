# How To Use

## Instructions
- Ensure you have all prerequisites
- Download release
- Run python file

## Arguments
- Input sound file:   Filepath to audio file (.wav supported)
- Input transcript:   Filepath to transcript
- Target output WPM:  The desired output WPM
- Dynamic range:      The maximum word-to-word speed increase in WPM (default 50)

## Pre-requisites to use release
- Python 1.13
- numpy
- nltk
- pickle
- soundfile
- [sentence transformers](https://huggingface.co/sentence-transformers)
- [forcealign](https://github.com/lukerbs/forcealign) --requires ffmpeg 7.1.1 ( windows: winget install "FFmpeg (Shared)" --version 7.1.1 )
- [torch](https://pytorch.org/)


## Datasets used:
  - [For Training the N-grams Model](https://www.kaggle.com/datasets/miguelcorraljr/ted-ultimate-dataset/data)
  - [For Measuring Sentence Similarity](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

