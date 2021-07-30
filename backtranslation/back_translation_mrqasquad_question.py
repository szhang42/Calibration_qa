from transformers import MarianMTModel, MarianTokenizer
import numpy as np
import json
import torch

enfr_model_name = 'Helsinki-NLP/opus-mt-en-zh'
enfr_tokenizer = MarianTokenizer.from_pretrained(enfr_model_name)
enfr_model = MarianMTModel.from_pretrained(enfr_model_name).cuda()
fren_model_name = 'Helsinki-NLP/opus-mt-zh-en'
fren_tokenizer = MarianTokenizer.from_pretrained(fren_model_name)
fren_model = MarianMTModel.from_pretrained(fren_model_name).cuda()

deen_model_name = 'Helsinki-NLP/opus-mt-de-en'
deen_tokenizer = MarianTokenizer.from_pretrained(deen_model_name)
deen_model = MarianMTModel.from_pretrained(deen_model_name).cuda()

ende_model_name = 'Helsinki-NLP/opus-mt-en-de'
ende_tokenizer = MarianTokenizer.from_pretrained(ende_model_name)
ende_model = MarianMTModel.from_pretrained(ende_model_name).cuda()


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
    translated = model.generate(**encoded, num_beams=2)

    # Convert the generated tokens indices back into text
    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
    
    return translated_texts

def back_translate_enfr(texts):# , source_lang="en", target_lang="fr"):
    # Translate from source to target language
    fr_texts = translate(texts, enfr_model, enfr_tokenizer,
            language='en')

    # Translate from target language back to source language
    back_translated_texts = translate(fr_texts, fren_model, fren_tokenizer,
            language='fr')

    return back_translated_texts

def back_translate_ende(texts):
    de_texts = translate(texts, ende_model, ende_tokenizer, language='en')
    return translate(de_texts, deen_model, deen_tokenizer, language='de')

input_file = open('SearchQA.jsonl', 'r')
output_file = open('backtranslated_question_searchqa_ende.json', 'w')

output_file.write(input_file.readline())
# json.dump(json.load(input_file.readline()), output_file)
count = 0
while True:
    data = input_file.readline()
    if len(data) <= 1:
        break
    data = json.loads(data)
    
    with torch.no_grad():
        number_quest = len(data['qas'])
        contexts = [data['qas'][index]["question"] for index in range(number_quest)]
        new_contexts = back_translate_ende(contexts)
        new_contexts = [new_contexts[index] if new_contexts[index] != '' else contexts[index] for index in range(len(new_contexts))]
        # new_contexts = [new_contexts[index] for index in range(len(new_contexts)) if new_contexts[index] != '' not contexts[index]]
        new_contexts = back_translate_ende(new_contexts) 
        for index in range(number_quest):
            data['qas'][index]["question"] = new_contexts[index]
        # data['context'] = '.'.join(back_translate(contexts))
    json.dump(data, output_file)
    output_file.write('\n')
    print(count);count += 1

# print(id_dict[key])
# en_texts = ['new york'] 
# aug_texts = back_translate(en_texts, source_lang="en", target_lang="fr")
# print(aug_texts)

