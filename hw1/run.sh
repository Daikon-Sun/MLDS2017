git clone git@gitlab.com:Daikon-Sun/MLDS_models.git
mv MLDS_models/data .
mv MLDS_models/logs .
mv MLDS_models/Training_Data .
rm -rf MLDS_models
python3 parse.py -sk -g data/glove.6B.300d.txt -t $1
python3 deprnn.py -me 0
mv submission.csv $2
