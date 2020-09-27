Instructions for running each task:
All tasks can be run on a python terminal by calling the respective files

Task 1: Input is param.json. It contains the input for dataset directory, resolution value, window length value and shift length value.
	The main output of the file will be stored in the same directory as the dataset. There will be a consolidated text file with all
	the words stored in the folder named 'Extras', within the 'Code' Folder

Task 2: Input is the consolidated file from Task 1. Make sure that task 1 is run successfully before running task 2.The output is a file 
	named 'vector.txt' stored in the same directory as the dataset.

Task 3: Input is the gesture file name and the values that need to be used (tf, tf-idf or tf-idf2). Make sure that task 1 is run before 
	running task 3. The output is a png file of the heatmap for the gesture with the selected values

Task 4: Input is the gesture file name and the values that need to be used (tf, tf-idf or tf-idf2). Make sure that task 1 is run before 
	running task 4. The output is displayed in the console where the program is run as a list containing ten gestures that are most 
	similar to the input gesture.

Read report found in folder 'Report' to understand the implementation and other information about the tasks.