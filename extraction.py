import re
import pandas as pd
import pysubs2

def extract_dialogues(read_path, save_path, max_distance=200, mode='“”'):
    '''
        read_path – to the file with the book text in txt format
        save_path – where to save the resulting dataset (overwrites the file)
        space – if two pieces of the direct speech are separated by this number of symbols,
                they will not be included in the final dataset
        mode – how direct speech is marked in the text;
                “” – "..." (<he said>, "...")
                ‘’ – ‘...’ (<he said>, ‘...’)
    '''
    assert mode in ['“”', '‘’']
    book = open(read_path, 'r')
        
    direct_speech = mode[0] + '.*?' + mode[1] + '(' + '((?!\n\n).)*' + mode[1] + ')?'
    authors_speech = '((?!\n\n).)*?'
    speech = direct_speech + '(' + authors_speech + direct_speech + ')?'
    speech = re.compile(speech, flags=re.DOTALL)
    save = []
    
    dialogues = speech.finditer(book.read())
    prev = next(dialogues)
    for curr in dialogues:
        if curr.start() - prev.end() <= max_distance:
            save.append(prev.group() + '\n' + curr.group())

        prev = curr
    
    pd.DataFrame({'text': save}).to_csv(save_path, index=False)
    
def extract_subtitles(read_path, save_path, max_distance=10):
    '''
        read_path – to the file with the subtitles in srt format
        save_path – where to save the resulting dataset (overwrites the file)
        max_distance – if two lines are more than this number of seconds apart,
                they will not be included in the final dataset
    '''
    subs = pysubs2.load(read_path)
    max_distance = pysubs2.make_time(s=max_distance)
    
    text = ''
    save = []
    
    prev = subs[0]
    for i in range(1, len(subs)):
        curr = subs[i] 
        if curr.start - prev.end <= max_distance:
            prev_text = prev.text.replace(r'\N', '\n')
            curr_text = curr.text.replace(r'\N', '\n')
            save.append(prev_text + '\n' + curr_text)
        prev = curr
        
    pd.DataFrame({'text': save}).to_csv(save_path, index=False)
    