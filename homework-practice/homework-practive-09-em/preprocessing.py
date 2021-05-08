from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from xml.etree.ElementTree import fromstring, ElementTree
import re

@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    sentence_pairs = []
    alignments = []
    getSentencePair = lambda x: [] if not x else list(map(lambda x: x.replace("*", "&"), x.split(' ')))
    getLabeledAlignment = lambda x: [] if not x else list(map(lambda x: (int(x.split('-')[0]), int(x.split('-')[1])), x.split(' ')))
    
    xml_file_handle = open(filename,'r')
    xml_as_string = xml_file_handle.read()
    xml_file_handle.close()
    xml_as_string = xml_as_string.replace("&", "*")
    xml_as_string = re.sub(u'[\x00-\x08\x0b\x0c\x0e-\x1F\uD800-\uDFFF\uFFFE\uFFFF]', '', xml_as_string)

    tree = ElementTree(fromstring(xml_as_string))    
    root = tree.getroot()
    for child in root:
        sentence_pair = SentencePair(source=getSentencePair(child[0].text), target=getSentencePair(child[1].text))
        alignment = LabeledAlignment(sure=getLabeledAlignment(child[2].text), possible=getLabeledAlignment(child[3].text))
        sentence_pairs.append(sentence_pair)
        alignments.append(alignment)

    return (sentence_pairs, alignments)


def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    """
    get_all_words = lambda x: np.unique(np.array([val for sublist in x for val in sublist]))

    source_sentences = [x.source for x in sentence_pairs]
    target_sentences = [x.target for x in sentence_pairs]
    source_words = get_all_words(source_sentences)
    target_words = get_all_words(target_sentences)

    source_dict = dict(enumerate(source_words, 0)) 
    source_dict = {source_dict[x]: x for x in source_dict}
    target_dict = dict(enumerate(target_words, 0))
    target_dict = {target_dict[x]: x for x in target_dict}
    
    if freq_cutoff is not None:
        source_cnt = {x: 0 for x in source_words}
        target_cnt = {x: 0 for x in target_words}
        for x in np.array([val for sublist in source_sentences for val in sublist]).flatten():
            source_cnt[x] = source_cnt[x] + 1
        for x in np.array([val for sublist in target_sentences for val in sublist]).flatten():
            target_cnt[x] = target_cnt[x] + 1
        
        source_words_sorted = source_words.tolist()
        #print(source_words_sorted)
        source_words_sorted.sort(key=lambda x: source_cnt[x])
        source_words_sorted = source_words_sorted[::-1]
        #print(source_words_sorted)

        target_words_sorted = target_words.tolist()
        target_words_sorted.sort(key=lambda x: target_cnt[x])
        target_words_sorted = target_words_sorted[::-1]
        
        if len(source_words_sorted) > freq_cutoff:
            source_words_sorted = source_words_sorted[0:freq_cutoff]
            #print(list(map(lambda x: source_cnt[x], source_words_sorted)))
        if len(target_words_sorted) > freq_cutoff:
            target_words_sorted = target_words_sorted[0:freq_cutoff]
            #print(list(map(lambda x: target_cnt[x], target_words_sorted)))
        
        source_dict = {x: source_dict[x] for x in source_words_sorted}
        target_dict = {x: target_dict[x] for x in target_words_sorted}
    
    return (source_dict, target_dict)
        
    


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    ans = []

    for sentence in sentence_pairs:
        good = True

        for x in sentence.source:
            if not x in source_dict:
                good = False
        
        for x in sentence.target:
            if not x in target_dict:
                good = False
        
        if good:
            source = np.array([source_dict[x] for x in sentence.source])
            target = np.array([target_dict[x] for x in sentence.target])
            ans.append(TokenizedSentencePair(source, target))

    return ans