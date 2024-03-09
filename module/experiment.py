import string
import nltk
import numpy as np
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

m = np.random.uniform(-1,1)
print(m)

# def tokenize_with_nltk(s : str):
#     s = s.strip().lower()
#     translator = str.maketrans('','', string.punctuation) 
#     words = nltk.word_tokenize(s)

#     tokenized_list = [w.translate(translator) for w in words if len(w.translate(translator)) > 0]    
#     return tokenized_list


# # Example usage:
# string_list = ["Hello, how are you?", "I'm doing well, thank you!"]
# tokenized_result = [tokenize_with_nltk(s) for s in string_list]
# # tokenized_result = [word.strip(string.punctuation) for word in string_list[1].split() if word.strip(string.punctuation).isalnum()]
 
# print(tokenized_result)