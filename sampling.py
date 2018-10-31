import xml.etree.ElementTree as ET
import bz2
import random

from itertools import chain

def sample_xml(filepath, sample_percent = 0.15, keep_attrs = ['lemma', 'msd', 'pos']):
    rows = []
    keep_sample = False
    total = 0
    replaced = 0
    with bz2.open(filepath, 'rb') as bf:
        for event, elem in ET.iterparse(bf, events=('start', 'end', 'start-ns', 'end-ns')):
            if event == "start" and elem.tag == "sentence":
                total += 1
                keep_sample = random.random() < sample_percent
                if keep_sample:
                    replaced += 1
                    sent_id = elem.get('id')
                    for word in elem.getiterator('w'):
                        r = {k: word.attrib[k] for k in keep_attrs}
                        r['sent_id'] = sent_id
                        r['word'] = word.text
                        rows.append(r)

    print("Total sentences: {}\tSampled sentences: {}".format(total, replaced))
    return rows

def reservoir_sample(filepath, number_rows, keep_attrs = ['lemma', 'msd', 'pos']):
    # Initialize the sample
    rows = [[]] * number_rows
    
    keep_sample = False
    i = 0
    replaced = 0
    with bz2.open(filepath, 'rb') as bf:
        for event, elem in ET.iterparse(bf, events=('start', 'end', 'start-ns', 'end-ns')):
            if event == "start" and elem.tag == "sentence":                
                j = random.randint(0, i)
                init_sample = i < number_rows
                replace_sample = i >= number_rows and j < number_rows
                keep_sample = init_sample or replace_sample
                if keep_sample:
                    if replace_sample:
                        # Reset the index and replace with new sentence
                        replaced += 1
                        rows[j] = []
                    sent_id = elem.get('id')
                    for word in elem.getiterator('w'):
                        r = {k: word.attrib[k] for k in keep_attrs}
                        r['sent_id'] = sent_id
                        r['word'] = word.text
                        if replace_sample:                                                        
                            rows[j].append(r)
                        else:
                            rows[i].append(r)                              
                i += 1

    print("Found total sentences: {}\t Replaced: {}".format(i, replaced))
    return i, list(chain(*rows))