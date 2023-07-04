"""Approximately simulates trec_eval using pytrec_eval."""

import argparse
import os
import sys

import pytrec_eval


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--qrel')
    parser.add_argument('--run')
    parser.add_argument('--tag')

    args = parser.parse_args()

    assert os.path.exists(args.qrel)
    assert os.path.exists(args.run)

    with open(args.qrel, 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)
    # for qid in qrel:
    #     for pid in qrel[qid]:
    #         if qrel[qid][pid] < 2:
    #             qrel[qid][pid] = 0


    with open(args.run, 'r') as f_run:
        run = pytrec_eval.parse_run(f_run)
    print(run)
    #   recip_rank, ndcg_cut.3
    # evaluator = pytrec_eval.RelevanceEvaluator(
    #     qrel, pytrec_eval.supported_measures)
    relevance_threshold = 2 if args.tag == 'cast20' else 1
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrel, {'ndcg_cut.3', 'recip_rank'}, relevance_level=relevance_threshold)
    
    results = evaluator.evaluate(run)

    def print_line(measure, scope, value):
        print('{:25s}{:8s}{:.4f}'.format(measure, scope, value))
        return '{:25s}{:8s}{:.4f}'.format(measure, scope, value)
    all_lines = []
    for query_id, query_measures in sorted(results.items()):
        for measure, value in sorted(query_measures.items()):
            # pass
            line = print_line(measure, query_id, value)
            all_lines.append(line+'\n')
    # print(all_lines)
    # print(results.keys())
    # Scope hack: use query_measures of last item in previous loop to
    # figure out all unique measure names.
    #
    # TODO(cvangysel): add member to RelevanceEvaluator
    #                  with a list of measure names.

    for measure in sorted(query_measures.keys()):
        line = print_line(
            measure,
            'all',
            pytrec_eval.compute_aggregated_measure(
                measure,
                [query_measures[measure]
                 for query_measures in results.values()]))
        
    with open('eval_result.txt', 'w') as f:
        for line in all_lines:
            f.write(line)

if __name__ == "__main__":
    sys.exit(main())