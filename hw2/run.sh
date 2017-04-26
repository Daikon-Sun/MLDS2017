git clone https://Daikon-Sun@gitlab.com/Daikon-Sun/MLDS_hw2_basic_model.git
cd MLDS_hw2_basic_model
if [[ "$1" = /* ]]
  then tid=$1 
  else tid="../"$1
fi
if [[ "$2" = /* ]]
  then feat=$2 
  else feat="../"$2
fi
python3 seq_to_seq.py -at -bi -ss -ed 4096 -me 0 -tid $tid -fp $feat
mv output.json ../
cd ..
