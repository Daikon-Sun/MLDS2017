git clone https://Daikon-Sun@gitlab.com/Daikon-Sun/MLDS_hw2_basic_model.git
cd MLDS_hw2_basic_model
if [[ "$1" != /* ]]; then $1 = "../"$1 fi
if [[ "$2" != /* ]]; then $2 = "../"$2 fi
python3 seq_to_seq.py -at -bi -ss -ed 4096 -me 0 -tid "../"$1 -fp "../"$2
mv output.json ../
cd ..
