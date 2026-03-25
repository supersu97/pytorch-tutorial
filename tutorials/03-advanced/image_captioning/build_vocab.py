import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO


class Vocabulary(object):
    """简单的词汇表包装器。"""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json, threshold):
    """构建一个简单的词汇表包装器。"""
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] 已对字幕进行分词。".format(i+1, len(ids)))

    # 如果词频小于 'threshold'，则丢弃该词。
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # 创建词汇表包装器并添加一些特殊标记。
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # 将词添加到词汇表中。
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main(args):
    vocab = build_vocab(json=args.caption_path, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("总词汇量: {}".format(len(vocab)))
    print("词汇表包装器已保存到 '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='data/annotations/captions_train2014.json', 
                        help='训练标注文件的路径')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', 
                        help='保存词汇表包装器的路径')
    parser.add_argument('--threshold', type=int, default=4, 
                        help='最小词频阈值')
    args = parser.parse_args()
    main(args)
