# MLDS 2017 hw1 - Language Model

## TODO
- parsing dependency tree
- training by RNN

## Parse
### Followings should be installed and versions is suggested:
- python 3.6.0
- spacy 1.6.0
- nltk 3.2.2
- `python -m spacy.en.download all` should be run first to download data for
training grammar model.(Probably forbidden for this homework)

### Usage
Running following command under `/hw1` directory.
```
$ ./parse.py
```
Make sure that training data is under `/hw1/Holmes_Training_Data/`.
Then, parsed dependency tree will be in `/hw1/Training_Data/`.

Memory usage is about `2GB`.
Time consumed is about `40 mins`.
