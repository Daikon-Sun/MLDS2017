import argparse
import re, string
import os, sys

argparser = argparse.ArgumentParser(description='Fill the testing with 5 potential answer')
argparser.add_argument('-i', '--input_file',
  type=str, default='testing_data.csv',
  help='INPUT_FILE is the testing data file to be converted'
       ' (default: %(default)s)')
argparser.add_argument('-o', '--output_file',
  type=str, default='testing_data_out.txt',
  help='OUTPUT_FILE is the output of data with filled sentences'
       ' (default: %(default)s)',)
argparser.set_defaults(comma_split=False)
args = argparser.parse_args()

with open(args.input_file,'r') as input_file:
  with open(args.output_file,'w') as output_file:
    next(input_file) # skip the first line
    lines = [l.strip('\n') for l in input_file.readlines()]
    for sentence in lines:
      words = sentence.split(',')
      new_line = ""
      for i in range(1, len(words) - 5): # There are 5 choices
        if i == 1:
          new_line = words[i]
        else:
          new_line = new_line + " , " + words[i]
      parts = new_line.split('_____') # 5 underline
      for i in range(5, 0, -1):
        output_file.write(parts[0] + words[len(words) - i] + parts[1] + "\n")
