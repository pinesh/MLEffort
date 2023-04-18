from pathlib import Path
import functools
import json

import tensorflow as tf

from main import model_fn
#["SUSE CaaS Platform 4.0", "SUSE Enterprise Storage 6", "", "SUSE Enterprise Storage 7", "SUSE Linux Enterprise Desktop 15 SP4", "SUSE Linux Enterprise High Performance Computing 15 SP4", "SUSE Linux Enterprise Server 15 SP4", "SUSE Linux Enterprise Server for SAP Applications 15 SP4", "SUSE Manager Proxy 4.3", "SUSE Manager Retail Branch Server 4.3", "SUSE Manager Server 4.3", "SUSE Linux Enterprise High Performance Computing 15 SP1-LTSS", "SUSE Linux Enterprise High Performance Computing 15 SP2-LTSS", "SUSE Linux Enterprise High Performance Computing 15 SP3-ESPOS", "SUSE Linux Enterprise High Performance Computing 15 SP3-LTSS", "SUSE Linux Enterprise Module for Desktop Applications 15 SP4", "SUSE Linux Enterprise Module for Package Hub 15 SP4", "SUSE Linux Enterprise Real Time 15 SP3", "SUSE Linux Enterprise Server 15 SP1-LTSS", "SUSE Linux Enterprise Server 15 SP2-LTSS", "SUSE Linux Enterprise Server 15 SP3-LTSS", "SUSE Linux Enterprise Server for SAP Applications 15 SP1", "SUSE Linux Enterprise Server for SAP Applications 15 SP2", "SUSE Linux Enterprise Server for SAP Applications 15 SP3", "SUSE Manager Proxy 4.1", "SUSE Manager Retail Branch Server 4.1", "SUSE Manager Server 4.1", "openSUSE Leap 15.4", "openSUSE Tumbleweed"
LINE = 'OpenSUSE SUSE OpenStack Cloud Crowbar 9'
DATADIR = '../data'
PARAMS = './results/params.json'
MODELDIR = './results/model'
'cpe:2.3:/a:cisco:webex_meetings:2106.1804.4106.5:-:{platform}'

def pretty_print(line, preds):
    words = line.strip().split()
    lengths = [max(len(w), len(p)) for w, p in zip(words, preds)]
    padded_words = [w + (l - len(w)) * ' ' for w, l in zip(words, lengths)]
    padded_preds = [p.decode() + (l - len(p)) * ' ' for p, l in zip(preds, lengths)]
    print('words: {}'.format(' '.join(padded_words)))
    print('preds: {}'.format(' '.join(padded_preds)))


def predict_input_fn(line):
    # Words
    words = [w.encode() for w in line.strip().split()]
    nwords = len(words)

    # Chars
    chars = [[c.encode() for c in w] for w in line.strip().split()]
    lengths = [len(c) for c in chars]
    max_len = max(lengths)
    chars = [c + [b'<pad>'] * (max_len - l) for c, l in zip(chars, lengths)]

    # Wrapping in Tensors
    words = tf.constant([words], dtype=tf.string)
    nwords = tf.constant([nwords], dtype=tf.int32)
    chars = tf.constant([chars], dtype=tf.string)
    nchars = tf.constant([lengths], dtype=tf.int32)

    return ((words, nwords), (chars, nchars)), None


if __name__ == '__main__':
    with Path(PARAMS).open() as f:
        params = json.load(f)

    params['words'] = str(Path(DATADIR, 'vocab.words.txt'))
    params['chars'] = str(Path(DATADIR, 'vocab.chars.txt'))
    params['tags'] = str(Path(DATADIR, 'vocab.tags.txt'))
    params['glove'] = str(Path(DATADIR, 'glove.npz'))

    estimator = tf.estimator.Estimator(model_fn, MODELDIR, params=params)
    predict_inpf = functools.partial(predict_input_fn, LINE)
    for pred in estimator.predict(predict_inpf):
        pretty_print(LINE, pred['tags'])
        break
