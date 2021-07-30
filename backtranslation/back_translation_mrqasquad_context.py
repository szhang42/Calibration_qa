from transformers import MarianMTModel, MarianTokenizer
import numpy as np
import json
import torch

target_model_name = 'Helsinki-NLP/opus-mt-en-de'
target_tokenizer = MarianTokenizer.from_pretrained(target_model_name)
target_model = MarianMTModel.from_pretrained(target_model_name).cuda()
en_model_name = 'Helsinki-NLP/opus-mt-de-en'
en_tokenizer = MarianTokenizer.from_pretrained(en_model_name)
en_model = MarianMTModel.from_pretrained(en_model_name).cuda()

number_seq = 10

def translate(texts, model, tokenizer, language="fr"):
    # Prepare the text data into appropriate format for the model
    template = lambda text: f"{text}" if language == "en" else f">>{language}<< {text}"
    src_texts = [template(text) for text in texts]

    # Tokenize the texts
    encoded = tokenizer.prepare_seq2seq_batch(src_texts)
    for key in encoded:
        encoded[key] = torch.from_numpy(np.array(encoded[key])).cuda()
    # Generate translation using model
    '''
    translated = model.generate(**encoded)

    # sents = tokenizer.batch_decode(model.generate(**encoded, no_repeat_ngram_size=2, num_beams=number_seq, num_return_sequences=number_seq))
    
    # return_lst = [i.replace('<pad>', '').strip() for i in sents]
    return return_lst
    '''
    translated = model.generate(**encoded)

    # Convert the generated tokens indices back into text
    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
    
    return translated_texts

def back_translate(texts, source_lang="en", target_lang="de"):
    # Translate from source to target language
    fr_texts = translate(texts, target_model, target_tokenizer, 
                         language=target_lang)

    # Translate from target language back to source language
    back_translated_texts = translate(fr_texts, en_model, en_tokenizer, 
                                      language=source_lang)
    
    return back_translated_texts

input_file = open('SQuAD.jsonl', 'r')
output_file = open('backtranslated_context_squda_ende.json', 'w')

output_file.write(input_file.readline())
# json.dump(json.load(input_file.readline()), output_file)
count = 0
while True:
    data = input_file.readline()
    if len(data) <= 1:
        break
    data = json.loads(data)
    
    with torch.no_grad():
        length = len(data['context'])
        contexts = [data['context'][index * length // 50 : length // 50 + index * length // 50] for index in range(length // 50 + 1)]
        # contexts = [item.strip() for item in data['context'].split('.') if item.strip() != '']
        # interval = len(contexts) // 3
        new_contexts = back_translate(contexts)#[:interval]) 
        # new_contexts += back_translate(contexts[interval:interval*2])
        # new_contexts += back_translate(contexts[interval*2:])
        data['context'] = ''.join(new_contexts)
    json.dump(data, output_file)
    output_file.write('\n')
    print(count);count += 1

# print(id_dict[key])
# en_texts = ['new york'] 
# aug_texts = back_translate(en_texts, source_lang="en", target_lang="fr")
# print(aug_texts)

