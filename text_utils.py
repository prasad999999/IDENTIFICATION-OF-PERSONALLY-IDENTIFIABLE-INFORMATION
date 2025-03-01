import re, json, nltk, itertools, spacy, difflib, math
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords
nltk_resources = ["punkt", "maxent_ne_chunker_tab", "words", "averaged_perceptron_tagger", "stopwords"]

for resource in nltk_resources:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource)

def string_tokenizer(text):
    final_word_list = []
    words_list = text.replace(" ", "\n").split("\n")
    
    for element in words_list: 
        if len(element) >= 2: 
            final_word_list.append(element)
    
    return final_word_list

def similarity(a, b): return difflib.SequenceMatcher(None, a, b).ratio() * 100

def get_regexes():
    with open('definitions.json', "r", encoding='utf-8') as json_file:
        _rules = json.load(json_file)
        return _rules

def email_pii(text, rules):
    email_rules = rules["Email"]["regex"]
    email_addresses = re.findall(email_rules, text)
    email_addresses = list(set(filter(None, email_addresses)))
    return email_addresses

def phone_pii(text, rules):
    phone_rules = rules["Phone Number"]["regex"]
    phone_numbers = re.findall(phone_rules, text)
    phone_numbers = list(itertools.chain(*phone_numbers))
    phone_numbers = list(set(filter(None, phone_numbers)))
    return phone_numbers

def id_card_numbers_pii(text, rules):
    results = []
    # Clear all non-regional regexes
    regional_regexes = {}
    for key in rules.keys():
        region = rules[key]["region"]
        if region is not None:
            regional_regexes[key]=rules[key]

    # Grab regexes from objects
    for key in regional_regexes.keys():
        region = rules[key]["region"]
        rule = rules[key]["regex"]
        
        try:
            match = re.findall(rule, text)
        except:
            match=[]

        if len(match) > 0:
            result = {'identifier_class':key, 'result': list(set(match))}
            results.append(result)

    return results

def regional_pii(text):
    import nltk
    from nltk import word_tokenize, pos_tag, ne_chunk
    from nltk.corpus import stopwords

    resources = ["punkt", "maxent_ne_chunker", "stopwords", "words", "averaged_perceptron_tagger"]

    try:
        nltk_resources = ["tokenizers/punkt", "chunkers/maxent_ne_chunker", "corpora/words.zip"]
        for resource in nltk_resources:
            if not nltk.data.find(resource): raise LookupError()
    except LookupError:
        for resource in resources:
            nltk.download(resource)

    stop_words = set(stopwords.words('english'))

    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    named_entities = ne_chunk(tagged_words)

    locations = []

    for entity in named_entities:
        if isinstance(entity, nltk.tree.Tree):
            if entity.label() in ['GPE', 'GSP', 'LOCATION', 'FACILITY']:
                location_name = ' '.join([word for word, tag in entity.leaves() if word.lower() not in stop_words and len(word) > 2])
                locations.append(location_name)

    return list(set(locations))

def keywords_classify_pii(rules, intelligible_text_list):
    scores = {}

    for key, rule in rules.items():
        scores[key] = 0
        keywords = rule.get("keywords", [])
        if keywords is not None:
            for intelligible_text_word in intelligible_text_list:
                for keywords_word in keywords:
                    if similarity(
                        intelligible_text_word.lower()
                            .replace(".", "")
                            .replace("'", "")
                            .replace("-", "")
                            .replace("_", "")
                            .replace(",", ""),
                        keywords_word.lower()
                    ) > 80: scores[key] += 1

    return scores

# if __name__ == '__main__':
#     original = """ THE UNION OF INDIA MAHARASHTRA STATE MOTOR DRIVING LICENCE DL No MHO3 200800000000 DOI 24-01-2007 Valid Till 23-01-2027 (NT) 09-03-2011 (TR) AED 15-03-2008 FOE{6 (2) O4WORCSESOH TO DRIVE FOLLOWING CLASS VEHICLES THROUGHOUT INDIA cOv DOI MCWG 24-01-2007 LMV 24-01-2007 TRANS 10-03-2008 DOB 01-12-1987 BG KHAN KHAN SIWd JA KHAN KHAN kd KKHANRAMAN NAGAR, BAIGANWADI; Govamdi, MUMBAI Ph 400M4} 'PBUtkRrv ID& SignaturefThumb E22tyRpc MahO} 2008261 Impression of Hokder Nane
# THE UNION OF INDIA MAHARASHTRA STATE MOTOR DRIVING LICENCE DL No MHO3 200800000000 DOI 24-01-2007 Valid Till 23-01-2027 (NT) 09-03-2011 (TR) AED 15-03-2008 FOE7{6 (2) O4WORCSESOH TO DRIVE FOLLOWING CLASS VEHICLES THROUGHOUT INDIA cOv DOI MCWG 24-01-2007 LMV 24-01-2007 TRANS 10-03-2008 DOB 01-12-1987 BG KHAN KHAN SIWd JA KHAN KHAN kd KKHANRAMAN NAGAR; BAIGANWADI; Govamdi MUMBAI Ph 400M4] 'PBUFkkrv ID& SignaturefThumb EotYRp MahO} 2008261 Impression of Hokder Nane
# THE UNION OF INDIA MAHARASHTRA STATE MOTOR DRIVING LICENCE DL No MHO3 200800000000 DOI : 24-01-2007 Valid Till 23-01-2027 (NT) 09-03-2011 (TR) AED 15-03-2008 FOE7{6 (2) O4WORCSESOH TO DRIVE FOLLOWING CLASS VEHICLES THROUGHOUT INDIA cOv DOI MCWG 24-01-2007 LMV 24-01-2007 TRANS 10-03-2008 DOB 01-12-1987 BG Name KHAN KHAN SIWd JA KHAN KHAN kd KKHANRAMAN NAGAR, BAIGANWADI Govamdi MUMBAI Ph 8 PBU Ewa rv 400M4] ID& SignaturefThumb BL2tYRUNDG MahO} 2008261 Impression of Hokder
# THE UNION OF INDIA MAHARASHTRA STATE MOTOR DRIVING LICENCE DL No Mk0} Zoo800o0d000 DOI 24-01-2007 Valid Till 23-01-2027 (NT) 09-01-7011 (TR} AED 15-03-2008 Fo8* {6 12} AUTHORISATON TO DRIVE FOLLOWNG CLASS OF VEHiCLeS THROUGHOUT INDIA COv DI MCWG 24-01-2007 LMv 24-01-007 TRANS 10-0+-2008 DOB 01-12.1987 BG MS MH Kiah Spmd#Khah Khan Md Kkhahrnam Mair UaioaNWAD, 13 RB [; T14 Snraluc aufoder Kvn
# B01 * THe UNION OF INDIA "#Karnanertextlrsh MAHARASHTRA STATE MotOR DRIVING LICENCE  Palid No MHo3 Zoo80oo00000 44DI 24-01-2007 23-01-2027, (NT) 5' 09- 03-7011 '8- AEO 15 0}-2008 338*i@ AUTHORISATON TODRIVE FOLLOWING CLASS OF VEHCLES THROUGHOUT INDIA COv_6 DOhW MCWG 24-01-2007_ LMV +#. 24-01-2007 TRANS 10-03-2008 DOB 01-12.1987 Kame KHAH KHAH SDWo:JAKHAH KHAN At KKHAHRAMAH NAGAR BAIGANWADI; Covaydl umbal  X9# E peu Fwa PN:Mpn EAENRaDs: wa 40%: Mho} 2008261 Spresyra{bypode; Ti _
# ATHE UNION OF INDIA Mgtal MAHARASHTRA STATE MOTOR DRIVING LICENCE; DL No MHOs 20080oooQ000 40 DOI ; 24-01-2007 Valid 23.012027  (NT 09-03-2011_ AED 4415-03-2008"' RoR3S AUTHORISATON TO DRIVE FOLLOWING CLASS OF.VEHCLES THROUGHOUT INDIA COvayr-DOW:N MCYG 24* 01-2007 _ LmVA  24-01*2007 TRANS 10.93.008' DOB 01*12*1987. Nano KHAN KHAN72 SpMO:JAKHAN KHAM _ Ad:KKHANRAMAM NAGAR; BAIGANWADI; Covnidl KumBAI: 8x03 PRQERR PIN uie EayRpS; Ifm IDJa Kaxo} 2008261 Signeturg Gupoder YX #2" 3 """
#     intelligible = string_tokenizer(original)
#     text = original
#     rules = get_regexes()

#     addresses = regional_pii(text)
#     emails = email_pii(text, rules)
#     phone_numbers = phone_pii(text, rules)

#     keywords_scores = keywords_classify_pii(rules, intelligible)
#     score = max(keywords_scores.values())
#     pii_class = list(keywords_scores.keys())[list(keywords_scores.values()).index(score)]

#     country_of_origin = rules[pii_class]["region"]
#     identifiers = id_card_numbers_pii(text, rules)

#     if score < 5:
#         pii_class = None

#     if len(identifiers) != 0:
#         identifiers = identifiers[0]["result"]

#     result = {
#         "pii_class" : pii_class,
#         "score" : score,
#         "country_of_origin": country_of_origin,
#         "identifiers" : identifiers,
#         "emails" : emails,
#         "phone_numbers" : phone_numbers,
#         "addresses" : addresses
#     }
#     print(result)
