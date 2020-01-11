import io
import re
import jieba
import multiprocessing
from gensim.models import Word2Vec

def train_douban():
    path = './dataset/raw/douban_single_turn.tsv'
    save_path = './word2vec/douban.word2vec.model'
    match_en = r'[a-zA-Z]+'
    sentences = []
    with io.open(path, encoding='UTF-8',mode='r') as file:
        for line in file:
            line = line.strip('\n')
            if re.match(match_en, line):
                pass
            else:
                line = line.split('\t')

                question = ['<start>']
                answer = ['<start>']

                # question += jieba.cut(line[0].rstrip().strip())
                # answer += jieba.cut(line[1].rstrip().strip())
                question += line[0].rstrip().strip().split(' ')
                answer += line[1].rstrip().strip().split(' ')
                question.append('<end>')
                answer.append('<end>')

                sentences.append(question)
                sentences.append(answer)
    model = Word2Vec(sentences=sentences, size=300, window=8, min_count=5, workers=multiprocessing.cpu_count())

    # save model
    model.save(save_path)


if __name__ == "__main__":
    train_douban()
