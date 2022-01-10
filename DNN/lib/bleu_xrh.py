#!/usr/bin/python
# -*- coding: UTF-8 -*-

from collections import *

import numpy as np

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu


class BleuScore:
    """
    评价指标 Bleu

    Author: xrh
    Date: 2019-12-20

    ref:
    1.BLEU: a Method for Automatic Evaluation of Machine Translation
    2.https://cloud.tencent.com/developer/article/1042161
    3.https://www.nltk.org/api/nltk.translate.html?highlight=bleu#module-nltk.translate.bleu_score

    """

    @staticmethod
    def get_ngrams(segment, N):
        """
        返回目标序列中, 所有 1_gram, 2_gram, ..., N_gram 的统计信息

        :param segment: 目标序列  ['the', 'cat', 'the', 'cat', 'on', 'the', 'mat']
        :param N: N_gram 的长度上限

        :return: count_n_gram

        {('the',): 3, ('cat',): 2, ('on',): 1, ('mat',): 1, ('the', 'cat'): 2, ('cat', 'the'): 1, ('cat', 'on'): 1, ('on', 'the'): 1, ('the', 'mat'): 1})

        """

        count_n_gram = Counter()

        for n in range(1, N + 1):  # n=1,2,...,N

            # 滑动窗口
            start = 0
            end = start + n

            while end <= len(segment):
                n_gram = tuple(segment[start:end])
                count_n_gram[n_gram] += 1

                start += 1
                end = start + n

        return count_n_gram

    @staticmethod
    def compute_bleu_corpus(reference_corpus, candidate_corpus, N=4, weight=None):
        """
        计算整个语料库的 bleu 分数

        语料库中有多个 机器翻译后的句子, 每一个机器翻译后的句子对应多条人工翻译的记录

        :param reference_corpus: 平行语料库人工翻译的记录

        eg. 3个 机器翻译的句子对应 3 组人工翻译的句子
        [
         [['1990', '09', '23'],],
         [['1990', '09', '23'],],
         [['1990', '09', '23'],]
        ]

        :param candidate_corpus: 机器的翻译结果

        eg. 一共 3个 机器翻译的句子
        [['1990', '09', '23'],
         ['2990', '09', '23'],
         ['9990', '09', '23']]

        :param N: N_gram 的长度上限,

              eg. N=2
              翻译结果: ['1990', '09', '23'] 只有 3个 term,
              计算 bleu 时设置 N_gram 的长度上限为 2( 仅考虑 1-gram, 2-gram 的加权和)

        :param weight: n_gram 对应的 precision 在计算 bleu 时的权重
        :return: bleu 分数

        """

        if weight == None:
            weight = np.array([1 / N] * N)

        candidate_length_sum = 0  # c
        reference_length_sum = 0  # r

        modified_n_gram_precision_list = []

        for (reference_list, candidate) in zip(reference_corpus, candidate_corpus):

            # (reference_list,translation) 一个 translation 和它对应的一组 reference

            candidate_ngram_count = BleuScore.get_ngrams(candidate, N)

            reference_ngram_count = Counter()

            min_length = float('inf')  # 依据论文, 选择最短的 reference 的长度计入r的求和

            for reference in reference_list:

                if len(reference) < min_length:
                    min_length = len(reference)

                reference_ngram_count |= BleuScore.get_ngrams(reference, N)  # 得到各个 n-gram 的 MaxRefCount

            reference_length_sum += min_length
            candidate_length_sum += len(candidate)  # candidate 的长度计入 c 的求和

            overlap = candidate_ngram_count & reference_ngram_count  # 得到各个 n-gram 的 count_clip

            # 计算 modified_n_gram_precision: p1 p2 p3 p4

            count_clip_n_gram_sum = np.zeros(N + 1)  # 统计 n_gram 的count_clip , n=1,2,3,4
            for k, v in overlap.items():
                count_clip_n_gram_sum[len(k)] += v

            count_n_gram_sum = np.zeros(N + 1)  # 统计 n_gram 的count , n=1,2,3,4
            for k, v in candidate_ngram_count.items():
                count_n_gram_sum[len(k)] += v

            modified_n_gram_precision = np.zeros(N + 1)  # n=1,2,3,4 ; N=4
            for n in range(1, N + 1):  # n=1,2,...,N

                if count_n_gram_sum[n] > 0 and count_clip_n_gram_sum[n] > 0:
                    modified_n_gram_precision[n] = count_clip_n_gram_sum[n] / count_n_gram_sum[n]

            modified_n_gram_precision_list.append(modified_n_gram_precision)

        ratio = reference_length_sum / candidate_length_sum  # r/c

        BP = 1  # brevity penalty

        if ratio >= 1:  # c <= r

            BP = np.exp(1 - ratio)

        bleu_score_list = []

        for i in range(len(candidate_corpus)):
            modified_n_gram_precision = modified_n_gram_precision_list[i][1:]

            bleu_score = BP * np.exp(np.dot(weight, np.log(modified_n_gram_precision)))

            bleu_score_list.append(bleu_score)

        return bleu_score_list


class Test:

    def test_get_ngrams(self):
        segment = ['the', 'cat', 'the', 'cat', 'on', 'the', 'mat']
        N = 2

        ngrams_count = BleuScore.get_ngrams(segment, N)

        print(ngrams_count)

    def test_compute_bleu_corpus(self):
        # test1
        reference = [[['this', 'is', 'small', 'test'], ['this', 'is', 'the', 'test']]]
        candidate = [['this', 'is', 'a', 'test']]

        score = BleuScore.compute_bleu_corpus(reference, candidate, N=2)
        print('test1:', score)

        score = corpus_bleu(reference, candidate, weights=(0.5, 0.5))
        print('nltk: ', score)

        # test2
        reference = [[['the', 'cat', 'is', 'on', 'the', 'mat'],
                      ['there', 'is', 'a', 'cat', 'on', 'the', 'mat']
                      ]]
        candidate = [['the', 'cat', 'the', 'cat', 'on', 'the', 'mat']]

        score = BleuScore.compute_bleu_corpus(reference, candidate, N=4)
        print('test2:', score)

        score = corpus_bleu(reference, candidate)
        print('nltk: ', score)

        # test3
        reference = [[['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]]
        candidate = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]

        score = BleuScore.compute_bleu_corpus(reference, candidate, N=4)
        print('test3:', score)

        score = corpus_bleu(reference, candidate)
        print('nltk: ', score)

        # test4
        reference = [[['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]]
        candidate = [['the', 'fast', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]

        score = BleuScore.compute_bleu_corpus(reference, candidate, N=4)
        print('test4:', score)

        score = corpus_bleu(reference, candidate)
        print('nltk: ', score)

        # test5
        reference = [[['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]]
        candidate = [['the', 'fast', 'brown', 'fox', 'jumped', 'over', 'the', 'sleepy', 'dog']]

        score = BleuScore.compute_bleu_corpus(reference, candidate, N=4)
        print('test5:', score)

        score = corpus_bleu(reference, candidate)
        print('nltk: ', score)

        # test6
        reference = [[['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]]
        candidate = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog', 'from', 'space']]

        score = BleuScore.compute_bleu_corpus(reference, candidate, N=4)
        print('test6:', score)

        score = corpus_bleu(reference, candidate)
        print('nltk: ', score)

        # test7
        reference = [[['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]]
        candidate = [['the', 'quick']]

        score = BleuScore.compute_bleu_corpus(reference, candidate, N=4)
        print('test7:', score)

        score = corpus_bleu(reference, candidate)
        print('nltk: ', score)


if __name__ == '__main__':
    test = Test()

    # test.test_get_ngrams()

    test.test_compute_bleu_corpus()
