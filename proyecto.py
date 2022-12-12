import pandas as pd
import spacy
import pickle
from sklearn.model_selection import train_test_split

#-----------------------------------Funciones---------------------------------------

def LematizadorOpiniones(opiniones,nlp):
    opiniones_lematizadas=[]
    for opinion in opiniones:
        if str(opinion)!=str("nan"):
            doc=nlp(opinion)
            opinion_tokenizada=[]
            opinion_lematizada=[]
            opinion_tokenizada=[token for token in doc]
            opinion_lematizada=[token.lemma_ for token in opinion_tokenizada]
            str_opinion_lematizada=""
            str_opinion_lematizada=" ".join(opinion_lematizada)
            opiniones_lematizadas.append(str_opinion_lematizada)
        else:
            opiniones_lematizadas.append(" ")
    return opiniones_lematizadas


def LematizadorTitulos(titles,nlp):
    titulos_lematizados=[]
    for titulo in titles:
        if str(titulo)!=str("nan"):
            doc=nlp(str(titulo))
            titulo_tokenizado=[]
            titulo_lematizado=[]
            titulo_tokenizado=[token for token in doc]
            titulo_lematizado=[token.lemma_ for token in titulo_tokenizado]
            str_titulo_lematizado=""
            str_titulo_lematizado=" ".join(titulo_lematizado)
            titulos_lematizados.append(str_titulo_lematizado)
        else: 
            titulos_lematizados.append(" ")
    return titulos_lematizados




#----------------------------------Inicio----------------------------------

#Esta parte solo es para tokenizar y lematizar el corpus y despues guardar los resultados en un pickle

# df = pd.read_excel('Rest_Mex_2022_Sentiment_Analysis_Track_Train.xlsx')

# titles = df['Title'].values
# opinions = df['Opinion'].values
# polarity=df['Polarity'].values
# attraction=df['Attraction'].values

# #Cargamos corpus de spacy para tokenizar y lematizar
# nlp=spacy.load('es_core_news_sm')


# #Tokenizamos y lemmatizamos opiniones
# opiniones_lematizadas=LematizadorOpiniones(opinions,nlp)
# titulos_lematizados=LematizadorTitulos(titles,nlp)

# #Guardamos opiniones y titulos lematizados en pickle

# matriz_lematizados=[]
# headers=[("Title"),("Opinion"),("Polarity")]
# matriz_lematizados.append(headers)
# for titulo_lematizado,opinion_lematizada,polaridad in zip(titulos_lematizados,opiniones_lematizadas,polarity):
#     matriz_lematizados.append([titulo_lematizado,opinion_lematizada,polaridad])


# #Guardamos matriz_lematizados en pickle
# with open("matriz_opiniones_titulos_lematizados.pickle", "wb") as f:
#     pickle.dump(matriz_lematizados, f)

with open("matriz_opiniones_titulos_lematizados.pickle", "rb") as f:
    obj = pickle.load(f)
df=pd.DataFrame(obj)

#Guardamos los datos de entrenamiento en X
X=[]
for i in range(len(obj)):
    if i!=0:
        x1=str(obj[i][0])
        x2=str(obj[i][1])
        X.append([x1.split(),x2.split()])

#Guardamos las etiquetas en y
y=[]
for i in range(len(obj)):
    if i!=0:
        y.append(obj[i][2])

#Dividimos en train y test
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2,random_state=0,shuffle= True)

#Carga del archivo txt a un dataframe para poder almacenarlo y convertirlo en diccionario
# df_dic = pd.read_csv('SEL_full.txt', sep=str("\t"), header=None, names=['Palabra', 'Nula[%]','Baja[%]', 'Media[%]', 'Alta[%]', 'PFA', 'Categoría'], skiprows=1, encoding='utf-8')
# df_dic.drop(['Nula[%]', 'Baja[%]', 'Media[%]', 'Alta[%]'], axis=1, inplace=True)

# #Convertimos a un diccionario
# diccionario = df_dic.set_index('Palabra').T.to_dict('list')

# #Guardamos el diccionario en un archivo .pkl
# with open("Dictionary.pickle", "wb") as tf:
#     pickle.dump(diccionario,tf)

with open("Dictionary.pickle", "rb") as f:
    dic = pickle.load(f)
    
#print(dic.get("abundancia")[0])
