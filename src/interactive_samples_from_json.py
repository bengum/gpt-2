#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf
from itertools import cycle
import re

import model, sample, encoder

def interact_model_from_json(
    model_name='124M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=1,
    models_dir='models',
    context_file=None,
    include_prefix=True,
    prefix='<|startoftext|>',
    truncate='<|endoftext|>',
    output_file=None
):
    """
    Interactively run the model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    :context_file : Path to json file containing context for each nsample. Will
     cycle through each item in file in format 
     [{'context': 'words words words'},
      {'context': 'more words words words'}]
    
      ...
     ]
    :include_prefix : to do
    :prefix : to do, usually <|startoftext|>
    :truncate : to do
    :output_file : Path to json file to hold results
    """
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    if context_file:
        with open(context_file, 'r', encoding='utf-8') as r:
            context_list = json.load(r)
    else:
        context_list = [' ']
    context_cycle = cycle(context_list)
    

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        

        # raw_text = input("Model prompt >>> ")
        # while not raw_text:
        #     print('Prompt should not be empty!')
        #     raw_text = input("Model prompt >>> ")
        # context_tokens = enc.encode(raw_text)
        
        output_list = []
        generated = 0
        for n in range(nsamples // batch_size):
            # Use the first value in the dict, endlessly cycling through list
            context_item = list(next(context_cycle).values())[0]
            context_tokens = enc.encode(context_item)
            out = sess.run(output, feed_dict={
                context: [context_tokens for _ in range(batch_size)]
            })[:, len(context_tokens):]
            for i in range(batch_size):
                generated += 1
                gen_text = enc.decode(out[i])
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                # print only between tags
                if prefix:
                    gen_text = enc.decode(context_tokens[:1]) + gen_text
                if truncate:
                    truncate_esc = re.escape(truncate)
                    if prefix and not include_prefix:
                        prefix_esc = re.escape(prefix)
                        pattern = '(?:{})(.*?)(?:{})'.format(prefix_esc,
                                                            truncate_esc)
                    else:
                        pattern = '(.*?)(?:{})'.format(truncate_esc)

                    trunc_text = re.search(pattern, gen_text, re.S)
                    if trunc_text:
                        gen_text = trunc_text.group(1)
                # Need to output to file in addition to printing
                print(context_item, gen_text)
                output_list.append({'text': gen_text, 
                                    'context': context_item,
                                    'index': n})
        print("=" * 80)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as w:
                json.dump(output_list, w)

if __name__ == '__main__':
    fire.Fire(interact_model_from_json)

