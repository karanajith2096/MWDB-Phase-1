import math

import Task_1 as t1

# Program to read the univariate data and calculate TF, TF-IDF and TF-IDF2 values

# Calculate the number of gesture files present in the directory
def get_num_gestures(words): 
    l = []
    for i in words.keys():
        w = i.split(",")
        gesture_id = w[0][1:]
        if gesture_id not in l:
            l.append(gesture_id)
    
    return len(l)

# To calculate TF values
def tf_values(words, list_set):   

    tf = {}
    for i in words.keys():
        sp = i.split(",")
        pair = "{"  + sp[0][1:] + "," + sp[1] + "}"
        
        if pair in tf.keys():
            tf[pair][list_set.index(words[i])] = tf[pair][list_set.index(words[i])] + 1
        else:
            c = []
            for i in range(len(list_set)):
                c.append(0)
            tf[pair] = c
        
    for i in tf.keys():
        total = sum(tf[i])
        tf[i] = [x / total for x in tf[i]]

    # f = open("tf_vectors.txt", "w")
    # for k, v in tf.items():
    #     f.write(str(k) + " " + str(v) + "\n")

    return tf

# To calculate IDF values
def idf_values(words, list_set):

    idf = {}
    g = {}
    n = get_num_gestures(words) # Number of gesture documents 
    for i in words.keys():
        sp = i.split(",")
        gesture_id = sp[0][1:]
        sensor_id = sp[1]
        pt = "g" + gesture_id + "," + "s" + sensor_id
        if sensor_id in idf.keys():
            if pt in g.keys():
                if words[i] in g[pt]:
                    continue
                else:
                    g[pt].append(words[i])
                    idf[sensor_id][list_set.index(words[i])] = idf[sensor_id][list_set.index(words[i])] + 1
            else:
                g[pt] = []
                g[pt].append(words[i])
                idf[sensor_id][list_set.index(words[i])] = idf[sensor_id][list_set.index(words[i])] + 1
        else:
            c = []
            for t in range(len(list_set)):
                c.append(0)
            idf[sensor_id] = c
            
            g[pt] = []
            g[pt].append(words[i])
            idf[sensor_id][list_set.index(words[i])] = idf[sensor_id][list_set.index(words[i])] + 1
    
    for i in idf.keys():
        l = []
        for x in idf[i]:
            if x > 0:
                l.append(math.log(n / x))
            else:
                l.append(0)
        idf[i] = l

    return idf

# To calculate IDF2 values
def idf2_values(words, list_set):
    
    idf2 = {}
    s = {}
    n = 20 # Number of sensors
    for i in words.keys():
        sp = i.split(",")
        gesture_id = sp[0][1:]
        sensor_id = sp[1]
        pt = gesture_id + "," + sensor_id
        if gesture_id in idf2.keys():
            if pt in s.keys():
                if words[i] in s[pt]:
                    continue
                else:
                    s[pt].append(words[i])
                    idf2[gesture_id][list_set.index(words[i])] = idf2[gesture_id][list_set.index(words[i])] + 1
            else:
                s[pt] = []
                s[pt].append(words[i])
                idf2[gesture_id][list_set.index(words[i])] = idf2[gesture_id][list_set.index(words[i])] + 1
        else:
            c = []
            for i in range(len(list_set)):
                c.append(0)
            idf2[gesture_id] = c
    
    for i in idf2.keys():
        l = []
        for x in idf2[i]:
            if x > 0:
                # l.append(math.log((n / x))) # if words occuring in each sensor are not considered unique
                l.append(math.log((n))) # If words occuring are unique to each sensor
            else:
                l.append(0)
        idf2[i] = l

    # f = open("idf2_vectors.txt", "w")
    # for k, v in idf2.items():
    #     f.write(str(k) + " " + str(v) + "\n")

    return idf2

# To calculate TF-IDF values
def calculate_tf_idf(tf, idf):
    
    tf_idf = {}
    for i in tf.keys():
        sp = i.split(",")
        sensor_id = sp[1][:-1]
        idf_list = idf[sensor_id]
        final = []
        for j in range(len(idf_list)):
            final.append(idf_list[j] * tf[i][j])
        tf_idf[i] = final
    
    return tf_idf

# To calculate TF-IDF2 values
def calculate_tf_idf2(tf, idf2):
    
    tf_idf2 = {}
    for i in tf.keys():
        sp = i.split(",")
        sensor_id = sp[1][:-1]
        idf2_list = idf2[sensor_id]
        final = []
        for j in range(len(idf2_list)):
            final.append(idf2_list[j] * tf[i][j])
        tf_idf2[i] = final
    
    # f = open("tf-idf2_vectors.txt", "w")
    # for k, v in tf_idf2.items():
    #     f.write(str(k) + " " + str(v) + "\n")

    return tf_idf2


if __name__ == "__main__":
    print("Starting Task 2:")

    data = t1.load_params()

    words = {} # Contains all the uni-variate word vectors

    print("Reading word vector file as input..........")
    with open("Extras/word_vector_dictionary.txt", "r") as f:
        for line in f:
            w = line.split(" ") # Extracting id and word from the document created
            # print(len(w))
            word_id = w[0]
            word = "[" + w[1][1:-1] # + w[2][:-1] + ")"
            for i in range(2, len(w)):
                word = word + ", " + w[i][:-1]
            
            words[word_id] = word

    # Creating a set of all words that have occured in the word vector file
    set_words = set(words.values())
    list_set = list(set_words)
    list_set = sorted(list_set)

    print("Calculating TF values........")
    tf = tf_values(words, list_set)
    idf = idf_values(words, list_set)
    idf2 = idf2_values(words, list_set)

    print("Calculating TF-IDF values........")
    tf_idf = calculate_tf_idf(tf, idf)

    print("Calculating TF-IDF2 values........")
    tf_idf2 = calculate_tf_idf2(tf, idf2)

    print("Generating 'vectors.txt' file...............")
    ges = {}
    for k, v in tf.items():
        sp = k.split(",")
        gesture_id = sp[0][1:]
        if gesture_id not in ges:
            ges[gesture_id] = {}
            ges[gesture_id]['TF_vector'] = []
            ges[gesture_id]['TF_vector'] = ges[gesture_id]['TF_vector'] + v
        else:
            ges[gesture_id]['TF_vector'] = ges[gesture_id]['TF_vector'] + v

    
    for k, v in tf_idf.items():
        sp = k.split(",")
        gesture_id = sp[0][1:]
        if gesture_id not in ges:
            ges[gesture_id] = {}
            ges[gesture_id]['TF-IDF_vector'] = []
            ges[gesture_id]['TF-IDF_vector'] = ges[gesture_id]['TF-IDF_vector'] + v
        else:
            if 'TF-IDF_vector' not in ges[gesture_id]:
                ges[gesture_id]['TF-IDF_vector'] = []
            ges[gesture_id]['TF-IDF_vector'] = ges[gesture_id]['TF-IDF_vector'] + v

    
    for k, v in tf_idf2.items():
        sp = k.split(",")
        gesture_id = sp[0][1:]
        if gesture_id not in ges:
            ges[gesture_id] = {}
            ges[gesture_id]['TF-IDF2_vector'] = []
            ges[gesture_id]['TF-IDF2_vector'] = ges[gesture_id]['TF-IDF2_vector'] + v
        else:
            if 'TF-IDF2_vector' not in ges[gesture_id]:
                ges[gesture_id]['TF-IDF2_vector'] = []
            ges[gesture_id]['TF-IDF2_vector'] = ges[gesture_id]['TF-IDF2_vector'] + v

    f = open(data['directory'] + "/vectors.txt", "w")
    for k, v in ges.items():
        f.write(k + ": {\n" + "        TF Vector: " + str(v['TF_vector']) + "\n        TF-IDF Vector: " + str(v['TF-IDF_vector']) + "\n        TF-IDF2 Vector: " + str(v['TF-IDF2_vector']) + "\n}\n")

    print("Task 2 completed")