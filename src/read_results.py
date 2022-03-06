import json
import pandas as pd
import argparse
from pathlib import Path


def main(path_prefix, save=True):
    path_prefix = Path(path_prefix)
    res = []
    for dirname in path_prefix.parent.iterdir():
        if not str(dirname.absolute()).startswith(str(path_prefix.absolute())):
            continue
        if args.infix is not None and not args.infix in str(dirname.absolute()):
            continue


        if str(dirname).endswith('pretrain'):
            results_file = dirname / 'all_results.json'
            with open(results_file, 'r') as f:
                result_dict = json.load(f)
            res.append({'name': dirname.name,
                        'perplexity': result_dict['perplexity']})
        else:
            results_file = dirname / 'eval_results.json'
            with open(results_file, 'r') as f:
                result_dict = json.load(f)
            curr_res = {'name': dirname.name,
                        'eval_loss': result_dict['eval_loss'],
                        'eval_accuracy': result_dict['eval_accuracy']}

            results_file = dirname / 'test_results.json'
            with open(results_file, 'r') as f:
                result_dict = json.load(f)
            curr_res.update({
                        'test_loss': result_dict['eval_loss'],
                        'test_accuracy': result_dict['eval_accuracy']})
            res.append(curr_res)


    df = pd.DataFrame(res)
    df = df.sort_values('name')
    print(df)

    if save:
        if args.infix is not None:
            save_path = Path(f'results/{path_prefix.name}_{args.infix}.tsv')
        else:
            save_path = Path(f'results/{path_prefix.name}.tsv')
        save_path.parent.mkdir(exist_ok=True, parents=True)

        df.to_csv(save_path, sep='\t', index=False)
        print(save_path)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate pretraining data')
    parser.add_argument('--prefix', type=str, help="prefix of output dirs", default=None)
    parser.add_argument('--infix', type=str, help="infix of output dirs", default=None)
    args = parser.parse_args()

    if args.prefix:
        main(args.prefix)
