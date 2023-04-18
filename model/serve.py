"""Reload and serve a saved model"""

from pathlib import Path
from tensorflow.contrib import predictor
from urllib.parse import quote_plus
import re

LINE = 'OpenSUSE SUSE OpenStack Cloud Crowbar 9'

class MLService:
    prediction_fn = None

    def __init__(self, export_dir):
        subdirs = [x for x in Path(export_dir).iterdir()
                   if x.is_dir() and 'temp' not in str(x)]
        latest = str(sorted(subdirs)[-1])
        self.prediction_fn = predictor.from_saved_model(latest)

    def serve(self, line):
        predictions = self.prediction_fn(parse_fn(line))
        print(self.interpret(self.stitch(line, predictions)))

    @staticmethod
    def stitch(line, predictions):
        sequence = line.split()
        preds = predictions['tags'][0]
        formatted_press = []
        if len(preds) != len(sequence):
            raise Exception('predictions did not match input sequence')

        for w, t in zip(sequence, preds):
            formatted_press.append({'word': w, 'tag': t})

        print(formatted_press)
        return formatted_press

    @staticmethod
    def interpret(p):
        s_type = '/a'
        vendor = ''
        product = ''
        version = ''
        sub = ''
        platform = ''

        # compose
        for elem in p:
            if elem['tag'] == b'B-VENDOR':
                vendor = elem['word']
            elif elem['tag'] == b'I-VENDOR':
                vendor += f"_{elem['word']}"
            elif elem['tag'] == b'B-PRODUCT':
                product = elem['word']
            elif elem['tag'] == b'I-PRODUCT':
                product += f"_{elem['word']}"
            elif elem['tag'] == b'B-VERSION':
                version = elem['word']
            elif elem['tag'] == b'B-PLATFORM':
                platform = elem['word']
            elif elem['tag'] == b'I-PLATFORM':
                platform += f" {elem['word']}"

        if len(platform) > 0:
            # drop words edition, for and track
            platform = re.sub(r'(?i)(for|edition|track)', '', platform)
            platform = re.sub(r'[\s+]', ' ', platform)
            platform = '~~~' + quote_plus(platform.strip().replace(' ', '_')) + '~~'
        cpe_total = f'cpe:2.3:{s_type}:{quote_plus(vendor)}:{quote_plus(product)}:{quote_plus(version)}:{quote_plus(sub)}:{quote_plus(platform)}'.lower()

        # Drop trailing colons
        return re.sub(r':+$', '', cpe_total)


def parse_fn(line):
    # Encode in Bytes for TF
    words = [w.encode() for w in line.strip().split()]

    # Chars
    chars = [[c.encode() for c in w] for w in line.strip().split()]
    lengths = [len(c) for c in chars]
    max_len = max(lengths)
    chars = [c + [b'<pad>'] * (max_len - l) for c, l in zip(chars, lengths)]

    return {'words': [words], 'nwords': [len(words)],
            'chars': [chars], 'nchars': [lengths]}


if __name__ == '__main__':
    e_dir = 'saved_model'
    ml_service = MLService(e_dir)
    ml_service.serve('OpenSUSE SUSE OpenStack Cloud Crowbar 9')
    ml_service.serve('Advanced Reports Project Advanced Reports 1.1.2 for SilverStripe')
    ml_service.serve('Air Sender Project Air Sender 1.0 for iPhone OS')
