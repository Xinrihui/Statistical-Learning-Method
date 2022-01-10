#!/usr/bin/python
# -*- coding: UTF-8 -*-

import re
import string
from bleu_xrh import *
from nltk.translate.bleu_score import corpus_bleu

class Evaluate:
    """
    评价翻译模型

    Author: xrh
    Date: 2021-9-26

    """
    def __init__(self, _null_str='<NULL>',
                       _start_str='<START>',
                       _end_str='<END>',
                       _unk_str='<UNK>'):
        """

        :param  _null_str: 空字符
        :param  _start_str: 句子的开始字符
        :param  _end_str: 句子的结束字符
        :param  _unk_str: 未登录字符
        """

        self._null_str = _null_str
        self._start_str = _start_str
        self._end_str = _end_str
        self._unk_str = _unk_str

        # 删除标点符号
        remove_chars = string.punctuation  # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
        remove_chars = remove_chars.replace("<", "")  # 不能删除 '<' , 因为 <START> 要作为一个单词
        remove_chars = remove_chars.replace(">", "")
        self.remove_chars_re = re.compile('[%s]' % re.escape(remove_chars))

        # 需要删除的控制词
        self.remove_word_re = re.compile(
            r'{}|{}|{}'.format(self._null_str, self._start_str, self._end_str, self._unk_str))


    def evaluate_bleu(self, references, candidates):
        """
        使用 bleu 对翻译结果进行评价

        :param references: 平行语料库的句子(人工翻译)

        eg. 3个 机器翻译的句子对应 3 组人工翻译的句子, 每一组有 2 个句子
          [
            ['<START> A group of race horses run down a track carrying jockeys . <END>', '<START> A horse race . <END>'],

            ['<START> A man and a woman kissing . <END>', '<START> A man and woman kissing in front of a crowd of people . <END>']

            ['<START> A man racing on a motorbike <END>', '<START> A motorcycle rider drives fast around a curve on a track . <END>']
         ]
        :param candidates: 机器翻译的句子

            candidates:
            [

             'riding horses <NULL> by beach <NULL> by <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>',

             'women in a park at the bottom of a set of a man and a girl with pink hair sitting on a rock near a row of red and white flowers <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> them',

             'racers ride their bikes <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL> <NULL>',

            ]
        :return:
        """

        references_arr = []
        candidates_arr = []

        assert len(references) == len(candidates)  # 机器翻译的结果个数和对照语料的组数要相同


        for candidate in candidates:  # 对待评价的候选句子进行清理

            # 删除 控制词
            candidate = self.remove_word_re.sub(' ', candidate)

            # 删除 标点符号
            candidate = self.remove_chars_re.sub(' ', candidate)

            candidate_split = candidate.split()

            candidates_arr.append(candidate_split)

        for reference_list in references:  # 对平行语料的句子进行清理

            group = []

            for reference in reference_list:

                # 删除 控制词
                reference = self.remove_word_re.sub(' ', reference)

                # 删除 标点符号
                reference = self.remove_chars_re.sub(' ', reference)

                reference_split = reference.split()

                group.append(reference_split)

            references_arr.append(group)

        bleu_score_dict_list = {}
        average_bleu_score_dict = {}

        # use xrh bleu
        bleu_score_dict_list['1-garm'] = BleuScore.compute_bleu_corpus(references_arr, candidates_arr,
                                                              N=1)
        bleu_score_dict_list['2-garm'] = BleuScore.compute_bleu_corpus(references_arr, candidates_arr,
                                                              N=2)

        # use nltk bleu
        # bleu_score_dict_list['1-garm'] = corpus_bleu(references_arr, candidates_arr,
        #                                                weights=(1.0, 0, 0, 0))
        # bleu_score_dict_list['2-garm'] = corpus_bleu(references_arr, candidates_arr,
        #                                                weights=(0.5, 0.5, 0, 0))

        average_bleu_score_dict['1-garm'] = np.average(bleu_score_dict_list['1-garm'])
        average_bleu_score_dict['2-garm'] = np.average(bleu_score_dict_list['2-garm'])

        return average_bleu_score_dict, bleu_score_dict_list
