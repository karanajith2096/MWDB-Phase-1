import seaborn as sb
import numpy as np
import Task_2 as t2
import matplotlib.pyplot as plt

# Program to generate heat maps for a selected gesture file

#Generate heatmap for gesture based on values
def gen_heatmap(ges_input, val_type, words, list_set):
    ges_values = {}
    ges_words = {}

    for key, value in val_type.items():
        w = key.split(",")
        gesture_id = w[0][1:]
        if gesture_id == ges_input:
            ges_values[w[1][:-1]] = value

    for key, value in words.items():
        w = key.split(",")
        gesture_id = w[0][1:]
        
        if gesture_id == ges_input:
            if w[1] in ges_words.keys():
                ges_words[w[1]].append(value)
            else:
                ges_words[w[1]] = []
                ges_words[w[1]].append(value)

    heat_map = np.zeros(shape=(len(ges_words), len(ges_words["1"])))

    for key in ges_words.keys():
        l = []
        for value in ges_words[key]:
            l.append(ges_values[key][list_set.index(value)])

        heat_map[int(key) - 1] = l

    heat = sb.heatmap(heat_map, cmap = "Greys")
    
    return heat

if __name__ == "__main__":

    print("Starting Task 3:")
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
    # print(list_set)

    tf = t2.tf_values(words, list_set)

    ges_val = input("Enter gesture file name: ")
    ty = input("Enter which value needs to be plotted: (tf, tf-idf, tf-idf2): ")

    print("Generating heatmap...........")
    if ty == "tf":
        heat = gen_heatmap(ges_val, tf, words, list_set)

        heat.set_ylabel('SENSOR_ID')
        heat.set_xlabel('TIME')
        heat.set_title(ty + " for " + ges_val)
        # plt.show()
        plt.savefig("../Outputs/" + ges_val + "_" + ty + ".png", bbox_inches='tight')
        print("Heatmap saved in 'Outputs' folder")
        print("Task 3 completed")

    elif ty == "tf-idf":
        idf = t2.idf_values(words, list_set)
        tf_idf = t2.calculate_tf_idf(tf, idf)
        heat = gen_heatmap(ges_val, tf_idf, words, list_set)

        heat.set_ylabel('SENSOR_ID')
        heat.set_xlabel('TIME')
        heat.set_title(ty + " for " + ges_val)
        # plt.show()
        plt.savefig("../Outputs/" + ges_val + "_" + ty + ".png", bbox_inches='tight')
        print("Heatmap saved in 'Outputs' folder")
        print("Task 3 completed")

    elif ty == "tf-idf2":
        idf2 = t2.idf2_values(words, list_set)
        tf_idf2 = t2.calculate_tf_idf2(tf, idf2)
        heat = gen_heatmap(ges_val, tf_idf2, words, list_set)

        heat.set_ylabel('SENSOR_ID')
        heat.set_xlabel('TIME')
        heat.set_title(ty + " for " + ges_val)
        plt.savefig("../Outputs/" + ges_val + "_" + ty + ".png", bbox_inches='tight')
        print("Heatmap saved in 'Outputs' folder")
        print("Task 3 completed")

    else:
        print("Input not registered")

    




