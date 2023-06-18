import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from bpt import BPTAttentionWrapperWithAlibi
import os

model_path_or_name = "bigscience/bloom-560m"
tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
model = AutoModelForCausalLM.from_pretrained(model_path_or_name)

factor = 1
max_positions = 64 * factor
model.config.max_position_embeddings=max_positions
tokenizer.model_max_length = max_positions

for each in model.transformer.h:
    each.self_attention = BPTAttentionWrapperWithAlibi(each.self_attention)

model = model.cuda().eval()

prompt = '''
Donald John Trump (born June 14, 1946) is an American politician, media personality, and businessman who served as the 45th president of the United States from 2017 to 2021.

Trump graduated from the Wharton School of the University of Pennsylvania with a bachelor's degree in 1968. He became president of his father's real estate business in 1971 and renamed it The Trump Organization. He expanded the company's operations to building and renovating skyscrapers, hotels, casinos, and golf courses and later started side ventures, mostly by licensing his name. From 2004 to 2015, he co-produced and hosted the reality television series The Apprentice. Trump and his businesses have been involved in more than 4,000 state and federal legal actions, including six bankruptcies.

Trump's political positions have been described as populist, protectionist, isolationist, and nationalist. He won the 2016 United States presidential election as the Republican nominee against Democratic nominee Hillary Clinton despite losing the national popular vote.[a] He became the first U.S. president with no prior military or government service. His election and policies sparked numerous protests. The 2017–2019 special counsel investigation established that Russia interfered in the 2016 election to favor the election of Trump. Trump promoted conspiracy theories and made many false and misleading statements during his campaigns and presidency, to a degree unprecedented in American politics. Many of his comments and actions have been characterized as racially charged or racist, and many as misogynistic.

Trump ordered a travel ban on citizens from several Muslim-majority countries, diverted military funding towards building a wall on the U.S.–Mexico border, and implemented a policy of family separations for apprehended migrants. He rolled back more than 100 environmental policies and regulations in an aggressive attempt to weaken environmental protections. Trump signed the Tax Cuts and Jobs Act of 2017 which cut taxes for individuals and businesses and rescinded the individual health insurance mandate penalty of the Affordable Care Act. He appointed 54 federal appellate judges and three United States Supreme Court justices. Trump initiated a trade war with China and withdrew the U.S. from the proposed Trans-Pacific Partnership trade agreement, the Paris Agreement on climate change, and the Iran nuclear deal. Trump met with North Korean leader Kim Jong-un three times, but made no progress on denuclearization. He reacted slowly to the COVID-19 pandemic, ignored or contradicted many recommendations from health officials in his messaging, and promoted misinformation about unproven treatments and the need for testing.

//
''' * factor

with torch.no_grad():
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    print(inputs.input_ids.shape)
    # we will use the below hacky util function to pass in the global mask, 
    # proper fix comes later, it needs to be a 2D list always, 
    # with batch size as the top level and the token index you want to attend to in the next
    inputs['repetition_penalty'] = 1.3
    inputs['no_repeat_ngram_size'] = True
    tokens = model.generate(**inputs, max_new_tokens = max_positions, use_cache = False)
    outputs = tokenizer.decode(tokens[0])
    print(outputs)
