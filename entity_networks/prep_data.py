"""
Loads and pre-processes the Children Book Test (CBT)  dataset into TFRecords.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import re
import json
import tarfile
import tensorflow as tf

from tqdm import tqdm

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('output_dir', '../CBT/data/records-tranctst', 'Dataset destination.')

SPLIT_RE = re.compile(r'(\W+)?')

PAD_TOKEN = '_PAD'
PAD_ID = 0
shortdataset = 2000
def tokenize(sentence):
    "Tokenize a string by splitting on non-word characters and stripping whitespace."
    return [token.strip().lower() for token in re.split(SPLIT_RE, sentence) if token.strip()]

def parse_stories(lines):
    """
    Parse the CBT format described here: https://research.fb.com/downloads/babi/
    """
    stories = []
    story = []
    #size of half window where for candidate c of length 1,  w + 1 + w = window size
    w = 0 
    for line in lines:
        line = line.decode('utf-8').strip()
        if not line:
            continue
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        #This is the line containing the query
        if nid == 21:
            query, answer, candidates = filter(None,line.split('\t'))
            answer = answer.lower()
            query = tokenize(query)
            candidates = [token.strip().lower() for token in candidates.split('|') if token.strip()]

	    substory = []
	    #Convert the stories to windows of 5 centered with candidate words.
	    if  w:
 		for sent in story:
                	n = len(sent)
			for i, x in enumerate(sent):
			    if x.lower() in candidates:
				s = sent[max(i-w,0):min(i+1+w,n)]
				substory.append(s)
            else:
            	substory = [x for x in story if x]
            stories.append((substory, query, answer, candidates))
            story.append('')
            #A limit just for making shorter datasets
            if  shortdataset and len(stories) >= shortdataset:
               break
        else:
            sentence = tokenize(line)
            story.append(sentence)

    return stories

def save_dataset(stories, path):
    """
    Save the stories into TFRecords.

    NOTE: Since each sentence is a consistent length from padding, we use
    `tf.train.Example`, rather than a `tf.train.SequenceExample`, which is
    _slightly_ faster.
    """
    writer = tf.python_io.TFRecordWriter(path)
    for story, query, answer, candidates in stories:
        story_flat = [token_id for sentence in story for token_id in sentence]

        story_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=story_flat))
        query_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=query))
        answer_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[answer]))
        candidates_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=candidates))
        features = tf.train.Features(feature={
            'story': story_feature,
            'query': query_feature,
            'answer': answer_feature,
            'candidates': candidates_feature,
        })

        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())
    writer.close()

def tokenize_stories(stories, token_to_id):
    "Convert all tokens into their unique ids."
    story_ids = []
    for story, query, answer, candidates in stories:
        story = [[token_to_id[token] for token in sentence] for sentence in story]
        query = [token_to_id[token] for token in query]
        answer = token_to_id[answer]
        candidates = [token_to_id[token] for token in candidates]
        story_ids.append((story, query, answer, candidates))
        if answer not in candidates:
              print(candidates)
    return story_ids

def get_tokenizer(stories):
    "Recover unique tokens as a vocab and map the tokens to ids."
    tokens_all = []
    for story, query, answer, candidates in stories:
        tokens_all.extend([token for sentence in story for token in sentence] + query + [answer] + candidates)
    vocab = [PAD_TOKEN] + sorted(set(tokens_all))
    token_to_id = {token: i for i, token in enumerate(vocab)}
    return vocab, token_to_id

def pad_stories(stories, max_sentence_length, max_story_length, max_query_length):
    "Pad sentences, stories, and queries to a consistence length."
    for story, query, _, candidates in stories:
        for sentence in story:
            for _ in range(max_sentence_length - len(sentence)):
                sentence.append(PAD_ID)
            assert len(sentence) == max_sentence_length

        for _ in range(max_story_length - len(story)):
            story.append([PAD_ID for _ in range(max_sentence_length)])

        for _ in range(max_query_length - len(query)):
            query.append(PAD_ID)
        for _ in range(10 - len(candidates)):
            candidates.append(PAD_ID)
        assert len(story) == max_story_length
        assert len(query) == max_query_length
        assert len(candidates) == 10

    return stories

def truncate_stories(stories, max_length):
    "Truncate a story to the specified maximum length."
    stories_truncated = []
    print("max_length ", max_length)
    for story, query, answer, candidates in stories:
        print("story len ", len(story))
	story_truncated = story[-max_length:]
        stories_truncated.append((story_truncated, query, answer, candidates))
    return stories_truncated

def main():
    "Main entrypoint."

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    task_names = [
        'CN',
        'NE',
        'P',
        'V',
    ]

    task_titles = [
        'Common Names',
        'Named Entities',
        'Prepositions',
        'Verbs',
    ]

    task_ids = [
        'CN',
        'NE',
        'P',
        'V',
    ]
    if not shortdataset:
    	task_sizes = [
       	 120769,
       	 108719,
       	 334030,
       	 105825,
    	]
    else:
	task_sizes = [shortdataset]*4

    for task_id, task_name, task_title, task_size in tqdm(zip(task_ids, task_names, task_titles, task_sizes), \
            desc='Processing datasets into records...'):
        stories_path_train = os.path.join('../CBT/data/','cbtest_' + task_name + '_train.txt')
        stories_path_test = os.path.join('../CBT/data/','cbtest_' + task_name + '_test_2500ex.txt')
        dataset_path_train = os.path.join(FLAGS.output_dir, task_id + '_train.tfrecords')
        dataset_path_test = os.path.join(FLAGS.output_dir, task_id + '_test.tfrecords')
        metadata_path = os.path.join(FLAGS.output_dir, task_id + '.json')

        truncated_story_length = 200

        f_train = open(stories_path_train)
        f_test = open(stories_path_test)

        stories_train = parse_stories(f_train.readlines())
        stories_test = parse_stories(f_test.readlines())

        stories_train = truncate_stories(stories_train, truncated_story_length)
        stories_test = truncate_stories(stories_test, truncated_story_length)

        vocab, token_to_id = get_tokenizer(stories_train + stories_test)
        vocab_size = len(vocab)

        stories_token_train = tokenize_stories(stories_train, token_to_id)
        stories_token_test = tokenize_stories(stories_test, token_to_id)
        stories_token_all = stories_token_train + stories_token_test

        story_lengths = [len(sentence) for story, _, _, _ in stories_token_all for sentence in story]
        max_sentence_length = max(story_lengths)
        max_story_length = max([len(story) for story, _, _, _ in stories_token_all])
        max_query_length = max([len(query) for _, query, _, _ in stories_token_all])

        with open(metadata_path, 'w') as f:
            metadata = {
                'task_id': task_id,
                'task_name': task_name,
                'task_title': task_title,
                'task_size': task_size,
                'max_query_length': max_query_length,
                'max_story_length': max_story_length,
                'max_sentence_length': max_sentence_length,
                'vocab': vocab,
                'vocab_size': vocab_size,
                'filenames': {
                    'train': os.path.basename(dataset_path_train),
                    'test': os.path.basename(dataset_path_test),
                }
            }
            json.dump(metadata, f)

        stories_pad_train = pad_stories(stories_token_train, \
            max_sentence_length, max_story_length, max_query_length)
        stories_pad_test = pad_stories(stories_token_test, \
            max_sentence_length, max_story_length, max_query_length)

        save_dataset(stories_pad_train, dataset_path_train)
        save_dataset(stories_pad_test, dataset_path_test)

if __name__ == '__main__':
    main()
