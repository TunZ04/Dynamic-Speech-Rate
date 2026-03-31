# How To Use

## Instructions
- Download release
- Ensure you have all prerequisites
- Run python file

## Arguments
- Input sound file:   Filepath to audio file (.wav supported)
- Input transcript:   Filepath to transcript
- Target output WPM:  The desired output WPM
- Dynamic range:      The maximum word-to-word speed increase in WPM (default 50)

## Pre-requisites
- Python 1.13
- numpy
- pandas
- nltk
- [huggingface](https://huggingface.co/)
- [forcealign](https://github.com/lukerbs/forcealign) --requires ffmpeg 7.1.1 ( windows: winget install "FFmpeg (Shared)" --version 7.1.1 )
- [torch](https://pytorch.org/)
- [audiotsm](https://github.com/Muges/audiotsm)
- soundfile
- [sentence_transformers](https://huggingface.co/docs/hub/sentence-transformers)


## Datasets used:
  - https://www.kaggle.com/datasets/miguelcorraljr/ted-ultimate-dataset/data
  - https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

