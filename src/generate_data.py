from pathlib import Path
from hmmlearn.hmm import MultinomialHMM
import numpy as np
import random
from functools import partial
from string import ascii_lowercase
from itertools import permutations
from tqdm import tqdm
import pickle
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer


def softmax(x, temp=1.0, axis=None):
    x /= temp
    if axis is None:
        x -= np.amax(x)
        return np.exp(x) / np.sum(np.exp(x))
    else:
        x -= np.expand_dims(np.amax(x, axis=axis), axis=axis)
        return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=axis), axis=axis)


def generate_hmm_parameters(
        n_components, n_symbols, perm_samples=10, transition_temp=1.0,
        emission_temp=1.0, start_temp=1.0):
    # generate parameters for HMM
    startprob = softmax(np.random.rand(n_components) - 0.5, start_temp)

    mixing = softmax(np.random.rand(perm_samples) - 0.5, transition_temp)
    mixing = mixing[:, np.newaxis, np.newaxis]
    perm_samples = [np.eye(n_components)[np.random.permutation(n_components)] for i in range(perm_samples)]
    transmat = np.sum(mixing * perm_samples, axis=0)

    emission_logits = np.random.rand(n_components, n_symbols) - 0.5
    # set transition to small prob
    emissionprob = softmax(emission_logits, emission_temp, axis=1)

    return startprob, transmat, emissionprob


def sample_from_hmm(hmm, length, seed=None):
    x, h = hmm.sample(n_samples=length, random_state=seed)
    return x.T[0], h


def get_default_sampler(hmm):
    return partial(sample_from_hmm, hmm=hmm)


def get_default_scorer(hmm):
    def score(x):
        proba = hmm.predict_proba([x])
        proba_last = proba[-1]
        proba_next_hidden = hmm.transmat_.T @ proba_last
        proba_next_emission = hmm.emissionprob_.T @ proba_next_hidden
        return proba_next_emission
    return score


def letter_generator(num):
    counter = 0
    for i in range(1, len(ascii_lowercase)):
        for perm in permutations(ascii_lowercase, i):
            yield ''.join(perm)
            counter += 1
            if counter >= num:
                return


def apply_vocab(tokens, vocab):
    return [vocab[tok] for tok in tokens]


def invert_vocab(tokens, vocab_to_int):
    return [vocab_to_int[tok] for tok in tokens]


def save_hmm_list(hmms, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(hmms, f)


def save_as_json(samples, save_path):
    df = pd.DataFrame(samples)
    df.to_json(save_path, orient='records', lines=True)


def generate_samples(num_samples, id_hmms, sample_length):
    id_samples = []
    for i in tqdm(range(num_samples)):
        j = np.random.choice(len(id_hmms))
        x, h = sample_from_hmm(id_hmms[j], sample_length)
        x = apply_vocab(x, vocab)
        id_samples.append({'text': ' '.join(x), 'hmm_idx': j, 'hiddens': h})
    return id_samples


def samples_to_raw(samples, out_path):
    with open(out_path, 'w') as f:
        for sample in samples:
            f.write(sample['text'] + ' ')


def load(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def create_downstream_examples(hmm_list, linear_weight, n_examples,
        example_len, save_path):

    def score(x):
        p_h1 = hmm.predict_proba(x[:, np.newaxis])[0]
        p_h0 = hmm.transmat_ @ p_h1
        return p_h0

    examples = []
    for i in tqdm(range(n_examples)):
        hmm = np.random.choice(hmm_list)
        x, h = sample_from_hmm(hmm, example_len)
        proba = score(x)
        logits = linear_weight @ proba
        label = np.argmax(logits)
        x = ' '.join(apply_vocab(x, vocab))

        x_posterior = hmm.emissionprob_.T @ proba
        examples.append({'sentence': x, 'label': label, 'hiddens': h, 'logits': logits, 'posterior': x_posterior})

    save_as_json(examples, save_path)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='generate pretraining data')
    parser.add_argument('--start_temp', type=float, default=10, help="probability of uniform delimiters")
    parser.add_argument('--transition_temp', type=float, default=0.5, help="probability of uniform delimiters")
    parser.add_argument('--emission_temp', type=float, default=0.2, help="probability of uniform delimiters")
    parser.add_argument('--n_symbols', type=int, default=100, help="number of symbols")
    parser.add_argument('--n_components', type=int, default=100, help="number of hidden states")
    parser.add_argument('--skip_resample', action='store_true', help="whether to re-sample the dataset")
    parser.add_argument('--n_examples', type=int, default=1500, help="number of pretraining sequences")
    parser.add_argument('--seed', type=int, default=1111, help="seed")
    parser.add_argument('--downstream_seed', type=int, default=None, help="seed for downstream task")
    parser.add_argument('--downstream_example_len', type=int, default=128, help="length of downstream task example")
    parser.add_argument('--downstream_seed_in_filename', action='store_true', help="downstream seed in filename")
    parser.add_argument('--data_dir', type=str, help="data directory")
    args = parser.parse_args()

    dataset_id = f'synthetic_trans{args.transition_temp}_emit{args.emission_temp}_start{args.start_temp}_nsymbols{args.n_symbols}_ncomponents{args.n_components}'
    if args.n_examples != 1500:
        dataset_id += f'_nexamples{args.n_examples}'
    dataset_id += f'_seed{args.seed}'

    data_dir = Path(args.data_dir)
    save_dir = data_dir / dataset_id
    save_dir.mkdir(exist_ok=True, parents=True)

    seed = args.seed
    np.random.seed(seed)
    random.seed(seed+2)
    n_components = args.n_components
    n_symbols = args.n_symbols
    n_perm_samples = n_components
    n_hmms = 1
    num_val_samples = 10
    num_train_samples = args.n_examples
    sample_length = 10240
    val_sample_length = 1024
    # downstream task num classes
    num_classes = 2
    n_downstream_train_examples = 5000
    n_downstream_val_examples = 500
    n_downstream_test_examples = 1000


    vocab = list(letter_generator(n_symbols))
    vocab = np.asarray(vocab)

    vocab_to_int = {k: i for i, k in enumerate(vocab)}

    if not args.skip_resample:
        hmm_list = []
        for i in range(n_hmms):
            startprob, transmat, emissionprob = generate_hmm_parameters(
                                                        n_components,
                                                        n_symbols,
                                                        perm_samples=n_perm_samples,
                                                        transition_temp=args.transition_temp,
                                                        emission_temp=args.emission_temp,
                                                        start_temp=args.start_temp)
            hmm = MultinomialHMM(n_components=n_components)
            hmm.startprob_ = startprob
            hmm.transmat_ = transmat
            hmm.emissionprob_ = emissionprob
            hmm_list.append(hmm)

        print("Generating samples")
        id_samples = generate_samples(num_train_samples, hmm_list, sample_length=sample_length)
        id_samples_val = generate_samples(num_val_samples, hmm_list, sample_length=val_sample_length)

        # save the hmm parameters for later verification
        save_hmm_list(hmm_list, save_dir / 'hmms.pkl')

        save_as_json(id_samples, save_dir / 'train.json')
        save_as_json(id_samples_val, save_dir / 'val.json')
        samples_to_raw(id_samples, save_dir / 'train.txt')
        samples_to_raw(id_samples_val, save_dir / 'val.txt')

        print("Generate Tokenizer")

        tokenizer_path = save_dir / 'tokenizer.json'
        tokenizer = Tokenizer(WordLevel(vocab=vocab_to_int, unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()

        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

        trainer = WordLevelTrainer(special_tokens=special_tokens)
        files = [save_dir / 'train.txt', save_dir / 'val.txt']
        files = [str(f) for f in files]
        tokenizer.train(files, trainer)
        tokenizer.save(str(tokenizer_path))

    else:
        hmm_list = load(save_dir / 'hmms.pkl')

    # Generate the examples for downstream task

    if args.downstream_seed is None:
        downstream_seed = args.seed
    else:
        downstream_seed = args.downstream_seed
        np.random.seed(downstream_seed)
        random.seed(downstream_seed+2)

    linear_weight = np.zeros(num_classes * n_components)
    idxs = np.random.choice(len(linear_weight), size=6)
    linear_weight[idxs] = np.random.randn(len(idxs))
    linear_weight = linear_weight.reshape((num_classes, n_components))

    if args.downstream_seed_in_filename:
        train_filename = save_dir / f'downstream_train_{downstream_seed}.json'
        val_filename = save_dir / f'downstream_val_{downstream_seed}.json'
        test_filename = save_dir / f'downstream_test_{downstream_seed}.json'
    else:
        train_filename = save_dir / 'downstream_train.json'
        val_filename = save_dir / 'downstream_val.json'
        test_filename = save_dir / 'downstream_test.json'


    create_downstream_examples(
            hmm_list,
            linear_weight,
            n_downstream_train_examples,
            args.downstream_example_len,
            train_filename)
    create_downstream_examples(hmm_list,
            linear_weight,
            n_downstream_val_examples,
            args.downstream_example_len,
            val_filename)
    create_downstream_examples(hmm_list,
            linear_weight,
            n_downstream_test_examples,
            args.downstream_example_len,
            test_filename)

    if args.downstream_seed_in_filename:
        linear_weight_filename = save_dir / f'linear_weight_{downstream_seed}.pkl'
    else:
        linear_weight_filename = save_dir / 'linear_weight.pkl'

    with open(linear_weight_filename, 'wb') as f:
        pickle.dump(linear_weight, f)

