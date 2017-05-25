git clone https://Daikon-Sun@gitlab.com/Daikon-Sun/MLDS_hw3_model.git
python3 generate_test.py -cf $1 -ds MLDS_hw3_model -md method_1 -gf MLDS_hw3_model/glove.6B.300d.txt
python3 generate.py -mp model_after_faces_epoch_595.ckpt -cvl 600 -md method_1 -ds MLDS_hw3_model
