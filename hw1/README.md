# MLDS 2017 hw1 - Language Model

## Parse
### Followings should be installed and versions is suggested:
- python 3.6.0
- spacy 1.6.0 (if dependency tree needed)
- nltk 3.2.2 (if dependency tree needed)
- `nltk.data english.pickle`
- `python -m spacy.en.download all` should be run first to download data for
training grammar model.(if dependency tree needed)

To download `nltk.data english.pickle`, go to a python shell:
```
>>> import nltk
>>> nltk.download()
>>> Downloader> d
Download which package (l=list; x=cancel)?
  Identifier> punkt
```
Then, the required data will be sucessfully downloaded.

### Usage
Running following command under `/hw1` directory.
```
$ ./parse.py
```
Make sure that training data is under `/hw1/Holmes_Training_Data/`.
Then, parsed sentences will be in `/hw1/Training_Data/`.

If dependency tree is needed, run following command under `/hw1`.
```
$ ./parse.py -d
```

More flexible options can be found by running
```
$ ./parse.py -h
```

Normal sentences
- Time consumed is about `16 mins`.

Dependency tree
- Memory usage is about `2 GB`.
- Time consumed is about `40 mins`.

## Rnn
Parameters are set to default value that performs well.
For customization, just run
```
$ ./deprnn.py -h
```
