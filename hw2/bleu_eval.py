#!/usr/bin/python3
import math, operator, sys
import numpy as np
import json
from functools import reduce
def count_ngram(candidate, references, n):
    clipped_count = 0
    count = 0
    r = 0
    c = 0
    for si in range(len(candidate)):
        # Calculate precision for each sentence
        ref_counts = []
        ref_lengths = []
        # Build dictionary of ngram counts
        for reference in references:
            ref_sentence = reference[si]
            ngram_d = {}
            words = ref_sentence.strip().split()
            ref_lengths.append(len(words))
            limits = len(words) - n + 1
            # loop through the sentance consider the ngram length
            for i in range(limits):
                ngram = ' '.join(words[i:i+n]).lower()
                if ngram in ngram_d.keys():
                    ngram_d[ngram] += 1
                else:
                    ngram_d[ngram] = 1
            ref_counts.append(ngram_d)
        # candidate
        cand_sentence = candidate[si]
        cand_dict = {}
        words = cand_sentence.strip().split()
        limits = len(words) - n + 1
        for i in range(0, limits):
            ngram = ' '.join(words[i:i + n]).lower()
            if ngram in cand_dict:
                cand_dict[ngram] += 1
            else:
                cand_dict[ngram] = 1
        clipped_count += clip_count(cand_dict, ref_counts)
        count += limits
        r += best_length_match(ref_lengths, len(words))
        c += len(words)
    if clipped_count == 0:
        pr = 0
    else:
        pr = float(clipped_count) / count
    bp = brevity_penalty(c, r)
    return pr, bp


def clip_count(cand_d, ref_ds):
    """Count the clip count for each ngram considering all references"""
    count = 0
    for m in cand_d.keys():
        m_w = cand_d[m]
        m_max = 0
        for ref in ref_ds:
            if m in ref:
                m_max = max(m_max, ref[m])
        m_w = min(m_w, m_max)
        count += m_w
    return count


def best_length_match(ref_l, cand_l):
    """Find the closest length of reference to that of candidate"""
    least_diff = abs(cand_l-ref_l[0])
    best = ref_l[0]
    for ref in ref_l:
        if abs(cand_l-ref) < least_diff:
            least_diff = abs(cand_l-ref)
            best = ref
    return best


def brevity_penalty(c, r):
    if c > r:
        bp = 1
    else:
        bp = math.exp(1-(float(r)/c))
    return bp


def geometric_mean(precisions):
    return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))


def BLEU():

    with open('output.json', 'r') as f:
      inputs = json.load(f)
    with open('MLDS_hw2_data/testing_public_label.json', 'r') as f:
      targets = json.load(f)
    targets = dict([[info['id'], info['caption']] for info in targets])

    scores = []
    for info in inputs:
      score = 0.0
      #print(info['caption'], info['id'])
      #print('-'*80)
      for cap in targets[info['id']]:
        #print(cap[:-1].lower(), end= '  -->>  ')
        candidate = [info['caption'].strip()]
        references = [[cap[:-1].lower().strip()]]
        precisions = []
        pr, bp = count_ngram(candidate, references, 1)
        #print(pr, bp)
        precisions.append(pr)
        score += geometric_mean(precisions) * bp
      scores.append(score/len(targets[info['id']]))
      #print(scores[-1])
      #print('='*80)
    print("BLEU SCORE: {}".format(np.mean(scores)))

BLEU()
