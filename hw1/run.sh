git clone git@gitlab.com:Daikon-Sun/MLDS_models.git
mv MLDS_models/data .
mv MLDS_models/logs .
rm -r MLDS_models
python3 deprnn.py -me 0
