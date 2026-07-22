import pickle

# Open the pickle file in binary read mode
with open('person_memory_20260713_153600.pkl', 'rb') as file:
    person_memory = pickle.load(file)

# Display the loaded data
print(person_memory)