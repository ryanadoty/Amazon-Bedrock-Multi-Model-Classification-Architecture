import pickle
# demonstrating how we can load the model into other files
model = pickle.load(open('../model.pkl', 'rb'))
# demonstrating how we can load the vectorizer into other files
vectorizer = pickle.load(open('../vectorizer.pkl', 'rb'))
# demonstrating how we can invoke the model we imported in other files
answer = model.predict(vectorizer.transform(["What is harrys last name?"]))
# testing the prediction of the model by checking its answer
print(answer[0])

