from pathlib import Path
import pandas as pd



def read_distances(path):
    res = []
    curr_res = {}
    with open(path, 'r') as f:
        run_summary_counter = 0
        for line in f:
            if 'Run summary' in line:
                run_summary_counter += 1
            if run_summary_counter == 2 and 'eval/tv_posterior' in line and len(res) == 0:
                toks = line.split('tv_posterior')
                curr_res['tv_posterior'] = float(toks[1].strip()) 
            if run_summary_counter == 2 and 'eval/l2_posterior' in line and len(res) == 0:
                toks = line.split('l2_posterior')
                curr_res['l2_posterior'] = float(toks[1].strip()) 
                res.append(curr_res)
                curr_res = {}

            if run_summary_counter == 3 and 'eval/tv_posterior' in line:
                toks = line.split('tv_posterior')
                curr_res['tv_posterior'] = float(toks[1].strip()) 
            if run_summary_counter == 3 and 'eval/l2_posterior' in line:
                toks = line.split('l2_posterior')
                curr_res['l2_posterior'] = float(toks[1].strip()) 
                res.append(curr_res)
                run_summary_counter = 0
    return res





def read_log(path):
    res = []
    curr_res = {}
    with open(path, 'r') as f:
        prev_line = None
        for line in f:
            if 'train metrics' in line:
                prev_toks = prev_line.split()
                curr_pathname = [tok for tok in prev_toks if 'output/' in tok][0]
                curr_name = Path(curr_pathname).parent.name
                curr_res['name'] = curr_name

            if 'eval_accuracy' in line and '>>' in line:
                if 'eval_acc' not in curr_res:
                    eval_acc = float(line.split('=')[1].strip())
                    curr_res['eval_acc'] = eval_acc * 100
                else:
                    test_acc = float(line.split('=')[1].strip())
                    curr_res['test_acc'] = test_acc * 100
                    res.append(curr_res)
                    curr_res = {}

            prev_line = line
    return res

