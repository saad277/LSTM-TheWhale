import spacy
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential,load_model
from keras.layers import Dense ,LSTM,Embedding
from keras.preprocessing.sequence import pad_sequences
from pickle import dump,load

def read_file(filepath):
    with open(filepath) as f:
        str_text=f.read();

    return str_text;


nlp=spacy.load("en",disable=["parser","trigger","ner"])

nlp.max_length=1198623;


def seperate_punc(doc_text):
    return [token.text.lower() for token in nlp(doc_text) if token.text not in "',#$\n~<=>/~--!@#$%^&*()-_{}|\;:? \n\n\n \n."  ]


d=read_file("./moby_dick_four_chapters.txt");


tokens=seperate_punc(d);



#passing #25 words and predicting number 26 word
train_length=25+1

text_sequences=[]

for i in range(train_length,len(tokens)):
   
    seq=tokens[i-train_length:i]
    text_sequences.append(seq)


type(text_sequences)

print(text_sequences[1])


tokenizer=Tokenizer();


tokenizer.fit_on_texts(text_sequences)


sequences=tokenizer.texts_to_sequences(text_sequences)

#tokenizer.index_Word
print(sequences[0])

vocabulary_size=len(tokenizer.word_counts)

print("vocabulary size ",vocabulary_size)


sequences=np.array(sequences)

print(sequences)

x=sequences[:,:-1];

y=sequences[:,-1]

y=to_categorical(y,num_classes=vocabulary_size+1)

seq_length=x.shape[1]

print("seq_length ",seq_length)

#LSTM MODEL

#use +1 

model=Sequential();
model.add(Embedding(vocabulary_size+1,seq_length,input_length=seq_length))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))

model.add(Dense(50,activation="relu"))
model.add(Dense(vocabulary_size+1,activation="softmax"))

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

#model.summary();



#model.fit(x,y,batch_size=128,epochs=2,verbose=1)


#model.save("my_LSTM_model.h5")

#saving tokenizer
dump(tokenizer,open("my_tokenizer","wb"))


def generate_text(model,tokenizer,seq_len,seed_text,num_gen_words):

    output_text=[]

    #25 words
    input_text=seed_text
        
    for i in range(num_gen_words):
        
        encoded_text=tokenizer.texts_to_sequences([input_text])[0]

        pad_encoded=pad_sequences([encoded_text],maxlen=seq_len,truncating="pre")

        pred_word_ind=model.predict_classes(pad_encoded,verbose=0)[0]

        pred_word=tokenizer.index_word[pred_word_ind]

        input_text+=" "+pred_word

        output_text.append(pred_word)
        
    return " ".join(output_text)
    

random_seed_text=text_sequences[0]

seed_text=" ".join(random_seed_text)

print(seed_text)

my_model=load_model("my_LSTM_model.h5");

#my_model=load_model("epochBIG.h5");

#tokenizer1=load(open("epochBIG","rb"))

print(my_model.summary())

x=generate_text(my_model,tokenizer,seq_length,seed_text,num_gen_words=25)

print(x)



