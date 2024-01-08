import math
import sys
import re
import os
import time
import numpy


# Function to get the arguments from user
def get_arguments():
    arguments = sys.argv
    if not (13 > len(arguments) > 1):
        print(
            "python ir_system.py --extract-collection <file> --query <query> --model \"bool\" --search-mode \"linear\" "
            "--documents \"original\" --stemming")

    file, query, model, search, documents, stemming = "", "", "bool", "linear", "original", False

    for i in range(len(arguments)):
        if arguments[i] == "--extract-collection":
            try:
                file = arguments[i + 1]
            except IndexError:
                print("Missing Parameter")
        elif arguments[i] == "--query":
            try:
                query = arguments[i + 1]
            except IndexError:
                print("Missing Parameter")
        elif arguments[i] == "--model":
            try:
                model = arguments[i + 1]
            except IndexError:
                print("Missing Parameter")
        elif arguments[i] == "--search-mode":
            try:
                search = arguments[i + 1]
            except IndexError:
                print("Missing Parameter")
        elif arguments[i] == "--documents":
            try:
                documents = arguments[i + 1]
            except IndexError:
                print("Missing Parameter")
        elif arguments[i] == "--stemming":
            try:
                stemming = True
            except IndexError:
                print("Missing Parameter")
    return file, query, model, search, documents, stemming


# Function to remove the stop words
def remove_stop_words(sentence, stop_word):
    pattern = r'\b{}\b'.format(re.escape(stop_word))
    modified_sentence = re.sub(pattern, '', sentence)
    pattern = r'\b{}\b'.format(re.escape(stop_word.capitalize()))
    modified_sentence = re.sub(pattern, '', modified_sentence)
    return modified_sentence


# Funtion to separate each fables from original file
def extraction(file):
    f = open(file, "r")
    original_text = f.read()
    split_text = original_text.split("\n\n\n\n")
    # fables CONTAINS THE FABLES
    fables = split_text[2:]
    # name_split CONTAINS THE NAME AND FABLE AT 0 AND 1
    name_split = []
    names = []
    for i in fables:
        name_split.append(i.split("\n\n\n"))
        names.append(
            f"{fables.index(i) + 1:02d}_{(name_split[fables.index(i)][0]).lower().lstrip().replace(' ', '_')}")
    label_write = open("labels.txt", "w")
    for i in range(len(names)):
        write_object = open(f"collection_original/{names[i]}.txt", "w")
        write_object.write(fables[i])
        label_write.write(f"{names[i]}\n")
        write_object.close()
    label_write.close()
    s = open("englishST.txt", "r")
    stop_words = s.read().split("\n")
    stop_less_fables = []
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for i in range(len(fables)):
        temp = fables[i]
        for l in punctuations:
            if l in fables[i]:
                temp = temp.replace(l, " ")
        for stop_word in stop_words:
            temp = remove_stop_words(temp, stop_word)
        stop_less_fables.append(temp)
    for i in range(len(names)):
        write_object = open(f"collection_no_stopwords/{names[i]}.txt", "w")
        write_object.write(stop_less_fables[i])
    return


# Function to select the document  (original / no stop words file)
def document_selection(documents):
    if documents == 'original':
        path = "collection_original/"
    else:
        path = "collection_no_stopwords/"

    return path


# Calculate precision and recall
def calculate_precision_recall(query_terms, relevant_docs, retrieved_docs):
    retrieved_docs = set(retrieved_docs)
    relevant_docs = set(relevant_docs)

    true_positives = len(relevant_docs.intersection(retrieved_docs))
    false_positives = len(retrieved_docs - relevant_docs)

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else '?'

    recall = true_positives / len(relevant_docs) if len(relevant_docs) > 0 else '?'

    return precision, recall


def process_ground_truth():
    ground_truth = {}

    with open("ground_truth.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            try:
                term, ids = line.strip().split(" - ")
                ids = [int(x) for x in ids.split(", ")]
                ground_truth[term] = ids
            except ValueError:
                # Skipping the lines that don't follow the expected format
                continue

    return ground_truth


def stem_things(query, text, search, model):
    if search == 'linear':
        query = " ".join(query.split())
        query = re.sub(r'[^\x20-\x7e]', '', query)
        query = query.split(" ")
        query = list(filter(None, query))[0]
        query = porter_stem(query)
        text = " ".join(text.split())
        text = re.sub(r'[^\x20-\x7e]', '', text)
        text = text.split(" ")
        text = list(filter(None, text))
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        for i in range(len(text)):
            for l in punctuations:
                if l in text[i]:
                    text[i] = text[i].replace(l, "")
            text[i] = porter_stem(text[i])
        text = ' '.join(text)

    elif model == 'vector':
        stemmed_query = []
        for q in query:
            q = " ".join(q.split())
            q = re.sub(r'[^\x20-\x7e]', '', q)
            q = q.split(" ")
            q = list(filter(None, q))[0]
            q = porter_stem(q)
            stemmed_query.append(q)
        query = stemmed_query
        temp = {}
        for i in text:
            if porter_stem(i) in temp:
                temp[porter_stem(i)].update(text[i])
            else:
                temp[porter_stem(i)] = set()
                temp[porter_stem(i)].update(text[i])
        text = temp
    elif search == 'inverted':
        query = " ".join(query.split())
        query = re.sub(r'[^\x20-\x7e]', '', query)
        query = query.split(" ")
        query = list(filter(None, query))[0]
        query = porter_stem(query)
        temp = {}
        for i in text:
            if porter_stem(i) in temp:
                temp[porter_stem(i)].update(text[i])
            else:
                temp[porter_stem(i)] = set()
                temp[porter_stem(i)].update(text[i])
        text = temp
    return query, text


# Function to search the word
def boolean_model(query, file_name, documents, search, stemming, model):
    if search == 'linear':
        path = document_selection(documents)
        read_fable = open(f"{path}{file_name}.txt", "r")
        text = read_fable.read()
        if stemming:
            query, text = stem_things(query, text, search, model)
        if re.search(r'\b{}\b'.format(query), text) or re.search(r'\b{}\b'.format(query.capitalize()), text):
            return file_name
        else:
            return ''
    elif search == 'inverted':
        if stemming:
            query, documents = stem_things(query, documents, search, model)
        if query in documents:
            return documents[query]
        else:
            return ''


# Function to get the list of documents where the search is found
def linear_search(query, model, documents, search, stemming):
    start_time = time.time()
    label_read = open("labels.txt", "r")
    label_read.seek(0)
    labels = list(filter(None, label_read.read().split("\n")))
    found_list = []
    if model == 'bool':
        if '&' in query:
            query_word_list = query.split('&')
            for query_word in query_word_list:
                temp = []
                for i in labels:
                    result = boolean_model(query_word_list[0], i, documents, search, stemming, model)
                    if result != '':
                        temp.append(result)
                if query_word_list.index(query_word) == 0:
                    found_list = list(sorted(set(found_list) | set(temp)))
                else:
                    found_list = list(sorted(set(found_list) & set(temp)))
        elif '|' in query:
            query_word_list = query.split('|')
            for query_word in query_word_list:
                temp = []
                for i in labels:
                    result = boolean_model(query_word_list[0], i, documents, search, stemming, model)
                    if result != '':
                        temp.append(result)
                # found_list = temp + found_list
                found_list = list(sorted(set(found_list) | set(temp)))
        elif '-' in query:
            query_word_list = list(filter(None, query.split('-')))
            temp = []
            for i in labels:
                result = boolean_model(query_word_list[0], i, documents, search, stemming, model)
                if result != '':
                    temp.append(result)
            for i in labels:
                if i not in temp:
                    found_list.append(i)
        else:
            for i in labels:
                result = boolean_model(query, i, documents, search, stemming, model)
                if result != '':
                    found_list.append(result)
    else:
        print("Model not available")
        sys.exit()
    if len(found_list) == 0:
        print("No documents found")
    else:
        print("Found Documents:")
    for i in found_list:
        print(i)
    end_time = int((time.time() - start_time) * 1000)
    found_list_ids = []
    for i in found_list:
        i = int(i[0:2])
        found_list_ids.append(i)
    try:
        if '&' in query:
            ground_truth = process_ground_truth()
            query_word_list = query.split('&')
            relevant_docs = []
            for i in range(0, len(query_word_list)):
                temp_relevant_docs = ground_truth.get(query_word_list[i], [])
                relevant_docs = relevant_docs + temp_relevant_docs
            relevant_docs = list(set(relevant_docs))
        elif '|' in query :
            query_word_list = query.split('|')
            ground_truth = process_ground_truth()
            relevant_docs = []
            for i in range(0, len(query_word_list)):
                temp_relevant_docs = ground_truth.get(query_word_list[i], [])
                relevant_docs = relevant_docs + temp_relevant_docs
            relevant_docs = list(set(relevant_docs))
        elif '-' in query :
            query_word_list = list(filter(None, query.split('-')))
            #query_word_list[0] = porter_stem((query_word_list[0]))
            ground_truth = process_ground_truth()
            ground_truth.pop(query_word_list[0])
            relevant_docs = []
            for i in ground_truth:
                temp_relevant_docs = ground_truth.get(i, [])
                relevant_docs = relevant_docs + temp_relevant_docs
            relevant_docs = list(set(relevant_docs))

        else :
          ground_truth = process_ground_truth()
          relevant_docs = ground_truth.get(query, [])
        precision, recall = calculate_precision_recall(query, relevant_docs, found_list_ids)
        print(f"T={end_time}ms,P=" + str(precision) + ",R=" + str(recall))
    except KeyError as e :
        print("Query word :" + str(query) + " doesn't exist in ground truth file, so the precision and recall cannot be calculated")

# function with implementation of inverted list
def inverted_list_function(query, model, documents, search, stemming):
    start_time = time.time()
    inverted_list = {}
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    folder_name = document_selection(documents)
    list_of_files = os.listdir(folder_name)
    for files in list_of_files:
        file_path = os.path.join(folder_name, files)
        with open(file_path, 'r') as opened_file:
            for line in opened_file:
                temp_words = line.split()
                for i in temp_words:
                    for l in punctuations:
                        if l in i:
                            i = i.replace(l, "")
                    i = i.lower()
                    if i in inverted_list:
                        inverted_list[i].add(files)
                    else:
                        inverted_list[i] = set()
                        inverted_list[i].add(files)
    label_read = open("labels.txt", "r")
    label_read.seek(0)
    labels = list(filter(None, label_read.read().split("\n")))
    matched_file_names = set()
    relevant_docs = []
    if model == 'bool':
        if '&' in query:
            ground_truth = process_ground_truth()
            query_word_list = query.split('&')
            relevant_docs = []
            for i in range(0, len(query_word_list)):
                temp_relevant_docs = ground_truth.get(query_word_list[i], [])
                relevant_docs = relevant_docs + temp_relevant_docs
            relevant_docs = list(set(relevant_docs))
            temp_matched_files = {}
            for query_word in query_word_list:
                temp_matched_files[query_word] = set()
                temp_matched_files[query_word] = [value for value in
                                                  boolean_model(query_word, labels, inverted_list,
                                                                search, stemming, model)]
            temp = temp_matched_files[query_word_list[0]]
            for entries in temp_matched_files:
                temp = set(set(temp_matched_files[entries]) & set(temp))
            matched_file_names = temp
        elif '|' in query:
            query_word_list = query.split('|')
            ground_truth = process_ground_truth()
            relevant_docs = []
            for i in range(0, len(query_word_list)):
                temp_relevant_docs = ground_truth.get(query_word_list[i], [])
                relevant_docs = relevant_docs + temp_relevant_docs
            relevant_docs = list(set(relevant_docs))
            for query_word in query_word_list:
                temp_matched_file_names = boolean_model(query_word, labels, inverted_list, search, stemming, model)
                for j in temp_matched_file_names:
                    matched_file_names.add(j)
        elif '-' in query:
            query_word_list = list(filter(None, query.split('-')))
            ground_truth = process_ground_truth()
            print("ground_truth:", ground_truth)
            ground_truth.pop(query_word_list[0])
            print("ground_truth:", ground_truth)
            relevant_docs = []
            for i in ground_truth:
                print(i)
                temp_relevant_docs = ground_truth.get(i, [])
                relevant_docs = relevant_docs + temp_relevant_docs
            relevant_docs = list(set(relevant_docs))
            not_matched_files = set()
            for query_word in query_word_list:
                temp_matched_file_names = boolean_model(query_word, labels, inverted_list, search, stemming, model)
                for j in temp_matched_file_names:
                    not_matched_files.add(j)
            for k in list_of_files:
                if k in not_matched_files:
                    continue
                else:
                    matched_file_names.add(k)
        else:
            query_word = query
            temp_matched_file_names = boolean_model(query_word, labels, inverted_list, search, stemming, model)
            for j in temp_matched_file_names:
                matched_file_names.add(j)
            ground_truth = process_ground_truth()
            relevant_docs = ground_truth.get(query, [])

    elif model == 'vector':
        matched_file_names = vector_space_model(query, inverted_list, stemming, search, model, documents)
    else:
        print("Model not available.")
    if len(matched_file_names) == 0:
        print("No documents found")
    else:
        print("Found Documents:")
    matched_file_names = sorted(matched_file_names)
    for i in matched_file_names:
        print(i.strip('.txt'))
    end_time = int((time.time() - start_time) * 1000)
    found_list_ids = []
    for i in list(matched_file_names):
        i = int(i[0:2])
        found_list_ids.append(i)
    try:
        if '&' in query:
            ground_truth = process_ground_truth()
            query_word_list = query.split('&')
            relevant_docs = []
            for i in range(0, len(query_word_list)):
                temp_relevant_docs = ground_truth.get(query_word_list[i], [])
                relevant_docs = relevant_docs + temp_relevant_docs
            relevant_docs = list(set(relevant_docs))
        elif '|' in query :
            query_word_list = query.split('|')
            ground_truth = process_ground_truth()
            relevant_docs = []
            for i in range(0, len(query_word_list)):
                temp_relevant_docs = ground_truth.get(query_word_list[i], [])
                relevant_docs = relevant_docs + temp_relevant_docs
            relevant_docs = list(set(relevant_docs))
        elif '-' in query :
            query_word_list = list(filter(None, query.split('-')))
            #query_word_list[0] = porter_stem((query_word_list[0]))
            ground_truth = process_ground_truth()
            ground_truth.pop(query_word_list[0])
            relevant_docs = []
            for i in ground_truth:
                temp_relevant_docs = ground_truth.get(i, [])
                relevant_docs = relevant_docs + temp_relevant_docs
            relevant_docs = list(set(relevant_docs))
        precision, recall = calculate_precision_recall(query, relevant_docs, found_list_ids)
        print(f"T={end_time}ms,P=" + str(precision) + ",R=" + str(recall))
    except KeyError as e :
        print("Query word :" + str(query) + " doesn't exist in ground truth file, so the precision and recall cannot be calculated")


# implements stemming rules
def porter_stem(word):
    # I read the porter.txt file and found your note.
    # Step 1a
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies"):
        word = word[:-2]
    elif word.endswith("ss"):
        word = word
    elif word.endswith("s"):
        word = word[:-1]

    # Step 1b
    if word.endswith("eed"):
        if len(word) > 4:
            word = word[:-1]
    elif re.search(r"[aeiou].*ed", word):
        word = re.sub(r"ed$", "", word)
        if re.search(r"at$|bl$|iz$", word):
            word += "e"
        elif re.search(r"[^aeiou](?:[^lsz]|[^aeiou]{2})$", word):
            word = word[:-1]
    elif re.search(r"[aeiou].*ing", word):
        word = re.sub(r"ing$", "", word)
        # if re.search(r"[^aeiou](?:[^lsz]|[^aeiou]{2})$", word):
        #    word = word[:-1]

    # Step 1c
    if re.search(r"[aeiou].*y", word):
        word = re.sub(r"y$", "i", word)

    # Step 2
    if re.search(r"[aeiou].*ational", word):
        word = re.sub(r"ational$", "ate", word)
    elif re.search(r"[aeiou].*tional", word):
        word = re.sub(r"tional$", "tion", word)
    elif re.search(r"[aeiou].*enci", word):
        word = re.sub(r"enci$", "ence", word)
    elif re.search(r"[aeiou].*anci", word):
        word = re.sub(r"anci$", "ance", word)
    elif re.search(r"[aeiou].*izer", word):
        word = re.sub(r"izer$", "ize", word)
    elif re.search(r"[aeiou].*abli", word):
        word = re.sub(r"abli$", "able", word)
    elif re.search(r"[aeiou].*alli", word):
        word = re.sub(r"alli$", "al", word)
    elif re.search(r"[aeiou].*entli", word):
        word = re.sub(r"entli$", "ent", word)
    elif re.search(r"[aeiou].*eli", word):
        word = re.sub(r"eli$", "e", word)
    elif re.search(r"[aeiou].*ousli", word):
        word = re.sub(r"ousli$", "ous", word)
    elif re.search(r"[aeiou].*ization", word):
        word = re.sub(r"ization$", "ize", word)
    elif re.search(r"[aeiou].*ation", word):
        word = re.sub(r"ation$", "ate", word)
    elif re.search(r"[aeiou].*ator", word):
        word = re.sub(r"ator$", "ate", word)
    elif re.search(r"[aeiou].*alism", word):
        word = re.sub(r"alism$", "al", word)
    elif re.search(r"[aeiou].*iveness", word):
        word = re.sub(r"iveness$", "ive", word)
    elif re.search(r"[aeiou].*fulness", word):
        word = re.sub(r"fulness$", "ful", word)
    elif re.search(r"[aeiou].*ousness", word):
        word = re.sub(r"ousness$", "ous", word)
    elif re.search(r"[aeiou].*aliti", word):
        word = re.sub(r"aliti$", "al", word)
    elif re.search(r"[aeiou].*iviti", word):
        word = re.sub(r"iviti$", "ive", word)
    elif re.search(r"[aeiou].*biliti", word):
        word = re.sub(r"biliti$", "ble", word)

    # Step 3
    if re.search(r"[aeiou].*icate", word):
        word = re.sub(r"icate$", "ic", word)
    elif re.search(r"[aeiou].*ative", word):
        word = re.sub(r"ative$", "", word)
    elif re.search(r"[aeiou].*alize", word):
        word = re.sub(r"alize$", "al", word)
    elif re.search(r"[aeiou].*iciti", word):
        word = re.sub(r"iciti$", "ic", word)
    elif re.search(r"[aeiou].*ful", word):
        word = re.sub(r"ful$", "", word)
    elif re.search(r"[aeiou].*ness", word):
        word = re.sub(r"ness$", "", word)

    # Step 4
    if re.search(r"[aeiou].*(al|ance|ence|er|ic|able|ible|ant|ement|ment|ent|ion|ou|ism|ate|iti|ous|ive|ize)$", word):
        word = re.sub(r"(al|ance|ence|er|ic|able|ible|ant|ement|ment|ent|ion|ou|ism|ate|iti|ous|ive|ize)$", "", word)

    # Step 5a
    if re.search(r"[aeiou].*e$", word):
        if len(word) > 1:
            word = word[:-1]

    # Step 5b
    if re.search(r"[aeiou].*ll$", word):
        if len(word) > 4:
            word = word[:-1]

    return word


# Vector Space model
def vector_space_model(query, inverted_list, stemming, search, model, documents):
    start_time = time.time()
    numpy.set_printoptions(threshold=sys.maxsize)
    label_read = open("labels.txt", "r")
    label_read.seek(0)
    labels = list(filter(None, label_read.read().split("\n")))
    N = len(labels)
    path = document_selection(documents)
    ground_truth = process_ground_truth()
    relevant_docs = ground_truth.get(query, [])
    if '|' in query:
        query_word_list = query.split('|')
        if stemming:
            query_word_list, inverted_list = stem_things(query_word_list, inverted_list, search, model)

        query_count_vector = []
        for i in inverted_list:
            query_count_vector.append(0)

        query_weight_vector = []
        for i in inverted_list:
            query_weight_vector.append(0.0)
        for j in query_word_list:
            try:
                k = list(inverted_list.keys()).index(j)
                query_count_vector[k] = query_count_vector[k] + 1
            except Exception as e:
                pass
        for j in query_word_list:
            try:
                k = list(inverted_list.keys()).index(j)
                tfqk = query_count_vector[k]
                max_tfqi = numpy.max(query_count_vector)
                nk = len(inverted_list[j])
                wqk = (.5 + (.5 * tfqk / max_tfqi)) * math.log10(N / nk)
                query_weight_vector[k] = wqk
            except Exception as e:
                pass
        document_word_count = {}
        document_word_weight = {}
        result_weight_vector = {}
        for name in labels:
            document_word_count[name] = []
            document_word_weight[name] = []
            result_weight_vector[name] = []
            for i in inverted_list:
                document_word_count[name].append(0)
            for i in inverted_list:
                document_word_weight[name].append(0.0)
            for i in inverted_list:
                result_weight_vector[name].append(0.0)
            read_fable = open(f"{path}{name}.txt", "r")
            text = read_fable.read()
            text_list = text.split()
            text_list = list(filter(None, text_list))
            punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
            for i in range(len(text_list)):
                for l in punctuations:
                    if l in text_list[i]:
                        text_list[i] = text_list[i].replace(l, "")
                text_list[i] = text_list[i].lower()
                if stemming:
                    text_list[i] = porter_stem(text_list[i])
            for m in text_list:
                try:
                    document_word_count[name][list(inverted_list.keys()).index(m)] += 1
                except Exception as e:
                    pass
            denominator = 0
            for m in text_list:
                try:
                    tf_di = document_word_count[name][list(inverted_list.keys()).index(m)]
                    ni = len(inverted_list[m])
                    denominator = denominator + math.pow((tf_di * math.log10(N / ni)), 2)
                except Exception as e:
                    pass
            for m in text_list:
                try:
                    tf_dk = document_word_count[name][list(inverted_list.keys()).index(m)]
                    nk = len(inverted_list[m])
                    numerator = tf_dk * math.log10(N / nk)
                    wdk = numerator / math.sqrt(denominator)
                    document_word_weight[name][list(inverted_list.keys()).index(m)] = wdk
                except Exception as e:
                    pass
            dot = numpy.dot(numpy.array(query_weight_vector),
                            numpy.array(document_word_weight[name]))
            result_weight_vector[name] = dot
        result_weight_vector = dict(sorted(result_weight_vector.items(), key=lambda x: x[1], reverse=True))
        matched_files = []
        for file, weight in result_weight_vector.items():
            if weight > 0.0:
                matched_files.append(file)
    elif '&' in query:
        query_word_list = query.split('&')
        if stemming:
            query_word_list, inverted_list = stem_things(query_word_list, inverted_list, search, model)
        query_count_dict = {}
        query_weight_dict = {}
        for j in query_word_list:
            query_count_dict[j] = []
            query_weight_dict[j] = []
            for i in inverted_list:
                query_count_dict[j].append(0)
            for i in inverted_list:
                query_weight_dict[j].append(0.0)
            for i in inverted_list:
                if i == j:
                    query_count_dict[j][list(inverted_list.keys()).index(i)] += 1
            for i in inverted_list:
                if i == j:
                    try:
                        k = list(inverted_list.keys()).index(j)
                        tfqk = query_count_dict[j][k]
                        max_tfqi = numpy.max(query_count_dict[j])
                        nk = len(inverted_list[j])
                        wqk = (.5 + (.5 * tfqk / max_tfqi)) * math.log10(N / nk)
                        query_weight_dict[j][k] = wqk
                    except Exception as e:
                        pass
        document_word_count = {}
        document_word_weight = {}
        result_weight_vector = {}
        for q in query_word_list:
            result_weight_vector[q] = {}
            for lbl in labels:
                result_weight_vector[q][lbl] = []
        for name in labels:
            document_word_count[name] = []
            document_word_weight[name] = []
            for i in inverted_list:
                document_word_count[name].append(0)
            for i in inverted_list:
                document_word_weight[name].append(0.0)
            read_fable = open(f"{path}{name}.txt", "r")
            text = read_fable.read()
            text_list = text.split()
            text_list = list(filter(None, text_list))
            punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
            for i in range(len(text_list)):
                for l in punctuations:
                    if l in text_list[i]:
                        text_list[i] = text_list[i].replace(l, "")
                text_list[i] = text_list[i].lower()
                if stemming:
                    text_list[i] = porter_stem(text_list[i])
            for m in text_list:
                try:
                    document_word_count[name][list(inverted_list.keys()).index(m)] += 1
                except Exception as e:
                    pass
            denominator = 0
            for m in text_list:
                try:
                    tf_di = document_word_count[name][list(inverted_list.keys()).index(m)]
                    ni = len(inverted_list[m])
                    denominator = denominator + math.pow((tf_di * math.log10(N / ni)), 2)
                except Exception as e:
                    pass
            for m in text_list:
                try:
                    tf_dk = document_word_count[name][list(inverted_list.keys()).index(m)]
                    nk = len(inverted_list[m])
                    numerator = tf_dk * math.log10(N / nk)
                    wdk = numerator / math.sqrt(denominator)
                    document_word_weight[name][list(inverted_list.keys()).index(m)] = wdk
                except Exception as e:
                    pass
            for j in query_word_list:
                result_weight_vector[j][name] = numpy.dot(numpy.array(query_weight_dict[j]),
                                                          numpy.array(document_word_weight[name]))
        matched_files = []
        temp_matched_files = {}
        for q in query_word_list:
            temp_matched_files[q] = []
            for file, weight in result_weight_vector[q].items():
                if weight > 0.0:
                    temp_matched_files[q].append(file)
                    if len(matched_files) == 0:
                        matched_files.append(file)
                    else:
                        matched_files = list(set(set(matched_files) & set(file)))
        temp = temp_matched_files[query_word_list[0]]
        for t in temp_matched_files:
            temp = set(set(temp_matched_files[t]) & set(temp))
        matched_files = list(temp)
    elif '-' in query:
        query_word_list = list(filter(None, query.split('-')))
        if stemming:
            query_word_list, inverted_list = stem_things(query_word_list, inverted_list, search, model)
        query_count_vector = []
        for i in inverted_list:
            query_count_vector.append(0)
        query_weight_vector = []
        for i in inverted_list:
            query_weight_vector.append(0.0)
        for j in query_word_list:
            try:
                k = list(inverted_list.keys()).index(j)
                query_count_vector[k] = query_count_vector[k] + 1
            except Exception as e:
                pass
        for j in query_word_list:
            try:
                k = list(inverted_list.keys()).index(j)
                tfqk = query_count_vector[k]
                max_tfqi = numpy.max(query_count_vector)
                nk = len(inverted_list[j])
                wqk = (.5 + (.5 * tfqk / max_tfqi)) * math.log10(N / nk)
                query_weight_vector[k] = wqk
            except Exception as e:
                pass
        document_word_count = {}
        document_word_weight = {}
        result_weight_vector = {}
        for name in labels:
            document_word_count[name] = []
            document_word_weight[name] = []
            result_weight_vector[name] = []
            for i in inverted_list:
                document_word_count[name].append(0)
            for i in inverted_list:
                document_word_weight[name].append(0.0)
            for i in inverted_list:
                result_weight_vector[name].append(0.0)
            read_fable = open(f"{path}{name}.txt", "r")
            text = read_fable.read()
            text_list = text.split()
            text_list = list(filter(None, text_list))
            punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
            for i in range(len(text_list)):
                for l in punctuations:
                    if l in text_list[i]:
                        text_list[i] = text_list[i].replace(l, "")
                text_list[i] = text_list[i].lower()
                if stemming:
                    text_list[i] = porter_stem(text_list[i])
            for m in text_list:
                try:
                    document_word_count[name][list(inverted_list.keys()).index(m)] += 1
                except Exception as e:
                    pass
            denominator = 0
            for m in text_list:
                try:
                    tf_di = document_word_count[name][list(inverted_list.keys()).index(m)]
                    ni = len(inverted_list[m])
                    denominator = denominator + math.pow((tf_di * math.log10(N / ni)), 2)
                except Exception as e:
                    pass
            for m in text_list:
                try:
                    tf_dk = document_word_count[name][list(inverted_list.keys()).index(m)]
                    nk = len(inverted_list[m])
                    numerator = tf_dk * math.log10(N / nk)
                    wdk = numerator / math.sqrt(denominator)
                    document_word_weight[name][list(inverted_list.keys()).index(m)] = wdk
                except Exception as e:
                    pass
            result_weight_vector[name] = numpy.dot(numpy.array(query_weight_vector),
                                                   numpy.array(document_word_weight[name]))
        result_weight_vector = dict(sorted(result_weight_vector.items(), key=lambda x: x[1], reverse=True))
        matched_files = []
        for file, weight in result_weight_vector.items():
            if weight == 0.0:
                matched_files.append(file)
    else:
        query_word_list = query.split(' ')
        if stemming:
            query_word_list, inverted_list = stem_things(query_word_list, inverted_list, search, model)

        query_count_vector = []
        for i in inverted_list:
            query_count_vector.append(0)
        query_weight_vector = []
        for i in inverted_list:
            query_weight_vector.append(0.0)
        for j in query_word_list:
            try:
                k = list(inverted_list.keys()).index(j)
                query_count_vector[k] = query_count_vector[k] + 1
            except Exception as e:
                pass
        for j in query_word_list:
            try:
                k = list(inverted_list.keys()).index(j)
                tfqk = query_count_vector[k]
                max_tfqi = numpy.max(query_count_vector)
                nk = len(inverted_list[j])
                wqk = (.5 + (.5 * tfqk / max_tfqi)) * math.log10(N / nk)
                query_weight_vector[k] = wqk
            except Exception as e:
                pass
        document_word_count = {}
        document_word_weight = {}
        result_weight_vector = {}
        for name in labels:
            document_word_count[name] = []
            document_word_weight[name] = []
            result_weight_vector[name] = []
            for i in inverted_list:
                document_word_count[name].append(0)
            for i in inverted_list:
                document_word_weight[name].append(0.0)
            for i in inverted_list:
                result_weight_vector[name].append(0.0)
            read_fable = open(f"{path}{name}.txt", "r")
            text = read_fable.read()
            text_list = text.split()
            text_list = list(filter(None, text_list))
            punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
            for i in range(len(text_list)):
                for l in punctuations:
                    if l in text_list[i]:
                        text_list[i] = text_list[i].replace(l, "")
                text_list[i] = text_list[i].lower()
                if stemming:
                    text_list[i] = porter_stem(text_list[i])
            for m in text_list:
                try:
                    document_word_count[name][list(inverted_list.keys()).index(m)] += 1
                except Exception as e:
                    pass
            denominator = 0
            for m in text_list:
                try:
                    tf_di = document_word_count[name][list(inverted_list.keys()).index(m)]
                    ni = len(inverted_list[m])
                    denominator = denominator + math.pow((tf_di * math.log10(N / ni)), 2)
                except Exception as e:
                    pass
            for m in text_list:
                try:
                    tf_dk = document_word_count[name][list(inverted_list.keys()).index(m)]
                    nk = len(inverted_list[m])
                    numerator = tf_dk * math.log10(N / nk)
                    wdk = numerator / math.sqrt(denominator)
                    document_word_weight[name][list(inverted_list.keys()).index(m)] = wdk
                except Exception as e:
                    pass
            dot = numpy.dot(numpy.array(query_weight_vector),
                            numpy.array(document_word_weight[name]))
            result_weight_vector[name] = dot
        result_weight_vector = dict(sorted(result_weight_vector.items(), key=lambda x: x[1], reverse=True))
        matched_files = []
        for file, weight in result_weight_vector.items():
            if weight > 0.0:
                matched_files.append(file)

    return matched_files


if __name__ == "__main__":
    file, query, model, search, documents, stemming = get_arguments()
    path1 = "collection_original/"
    path2 = "collection_no_stopwords/"
    if not os.path.exists(path1):
        os.makedirs(path1)
    if not os.path.exists(path2):
        os.makedirs(path2)
    if file != '':
        extraction(file)
    if query != '':
        if search == 'linear':
            linear_search(query, model, documents, search, stemming)
        elif model == 'vector':
            search = 'inverted'
            inverted_list_function(query, model, documents, search, stemming)
        elif search == 'inverted':
            inverted_list_function(query, model, documents, search, stemming)
        else:
            print("Search type not available")
