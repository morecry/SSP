"""Approximately simulates trec_eval using pytrec_eval."""

import argparse
from audioop import avg
import os
import sys
from scipy import stats

import pytrec_eval

def get_result(run, qrel, args):
    relevance_threshold = 2 if args.tag == 'cast20' else 1
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrel, {'ndcg_cut.3', 'recip_rank'}, relevance_level=relevance_threshold)
    
    results = evaluator.evaluate(run)
    ndcg_cut_3 = []
    recip_rank = []

    for qid in results:
        recip_rank.append(results[qid]['recip_rank'])
        ndcg_cut_3.append(results[qid]['ndcg_cut_3'])
    return recip_rank, ndcg_cut_3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qrel')
    parser.add_argument('--run1')
    parser.add_argument('--run2')
    parser.add_argument('--tag')

    args = parser.parse_args()

    assert os.path.exists(args.run1)
    assert os.path.exists(args.run2)

    with open(args.qrel, 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)
        
    with open(args.run1, 'r') as f_run1:
        run1 = pytrec_eval.parse_run(f_run1)

    with open(args.run2, 'r') as f_run2:
        run2 = pytrec_eval.parse_run(f_run2)

    
    run1_recip_rank, run1_ndcg_cut_3 = get_result(run1, qrel, args)
    run2_recip_rank, run2_ndcg_cut_3 = get_result(run2, qrel, args)
    # for i in range(len(run1_recip_rank)):
    #     print(run1_recip_rank[i] - run2_recip_rank[i])
    # print(run1_recip_rank - run2_recip_rank)
    # run1_recip_rank = [float('%.3f' % item) for item in run1_recip_rank]
    # run1_ndcg_cut_3 = [float('%.3f' % item) for item in run1_ndcg_cut_3]

    # run2_recip_rank = [float('%.3f' % item) for item in run2_recip_rank]
    # run2_ndcg_cut_3 = [float('%.3f' % item) for item in run2_ndcg_cut_3]



    print(sum(run2_recip_rank)/len(run2_recip_rank))
    print(sum(run1_recip_rank)/len(run1_recip_rank))
    print(sum(run2_ndcg_cut_3)/len(run1_ndcg_cut_3))
    print(sum(run1_ndcg_cut_3)/len(run1_ndcg_cut_3))
    # print(stats.levene(run1_recip_rank, run2_recip_rank))
    print(stats.ttest_rel(run1_ndcg_cut_3,run2_ndcg_cut_3,alternative='greater'))
    print(stats.ttest_rel(run1_recip_rank,run2_recip_rank,alternative='greater'))

if __name__ == "__main__":
    sys.exit(main())