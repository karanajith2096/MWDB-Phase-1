# Program that takes a gesture file as input and returns the 10 most similar gestures as output based on their values
import Task_2 as t2
import collections

# Get all the words for the query gesture
def get_ges_words(ges_val, tf):
    ges_values = {}
    for key, value in tf.items():
        w = key.split(",")
        gesture_id = w[0][1:]
        if gesture_id == ges_val:
            ges_values[w[1][:-1]] = value

    # print(ges_values)
    return ges_values

# Return list of 10 most similar gestures to query gesture
def get_sim_ges(ges_name, tf):
    closest = []
    completed = []
    ges_val = get_ges_words(ges_name, tf)
    for key in tf.keys():
        w = key.split(",")
        gesture_id = w[0][1:]

        if gesture_id not in completed:
            completed.append(gesture_id)
        else:
            continue

        other = get_ges_words(gesture_id, tf)
        difference = 0

        for key in ges_val.keys():
            for i in range(len(ges_val[key])):
                difference = difference + abs(ges_val[key][i] - other[key][i])

        # if ges_name == gesture_id:
        #     difference = 0

        if difference != 0:            
            similarity = 1 / difference
        elif difference == 0:
            similarity = 10

        l = []
        l.append(gesture_id)
        l.append(similarity)
        if len(closest) < 10:
            closest.append(l)
            closest = sorted(closest, key=lambda kv: kv[1], reverse=True)
        elif len(closest) == 10:
            if closest[-1][1] < l[1]:
                closest[-1] = l
                closest = sorted(closest, key=lambda kv: kv[1], reverse=True)
    
    final = []
    for i in closest:
        final.append(i[0])

    print(final)

if __name__ == "__main__":
    print("Starting Task 4:")

    ges_val = input("Enter gesture file name: ")
    
    words = {} # Contains all the uni-variate word vectors

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

    tf = t2.tf_values(words, list_set)

    ty = input("Enter which value needs to be plotted: (tf, tf-idf, tf-idf2): ")

    print("Generating list of 10 most similar gestures to '" + ges_val + "' for '" + ty + "' values:")
    if ty == "tf":
        get_sim_ges(ges_val, tf)
        print("Task 4 completed")

    elif ty == "tf-idf":
        idf = t2.idf_values(words, list_set)
        tf_idf = t2.calculate_tf_idf(tf, idf)
        get_sim_ges(ges_val, tf_idf)
        print("Task 4 completed")

    elif ty == "tf-idf2":
        idf2 = t2.idf2_values(words, list_set)
        tf_idf2 = t2.calculate_tf_idf2(tf, idf2)
        get_sim_ges(ges_val, tf_idf2)
        print("Task 4 completed")

    else:
        print("Input not registered")

    