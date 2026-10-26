import streamlit as st

#On lance avec streamlit run app.py

#1.1

st.title("Approximation de Taylor")

st.write("Bienvenue dans votre outil de visualisation.")

st.subheader("Ceci est une application permettant d'approcher une fonction à l'ordre de vos souhaits.")

#1.2
st.latex("P_n(x)=\sum_{k=0}^n\dfrac{f^{(k)}(0)}{k!}x^k")


#4.1

import streamlit as st
import autograd.numpy as np
from autograd import grad
from math import factorial
import matplotlib.pyplot as plt

def taylor1(X, derivatives):
    def taylor_intermed(x): #On crée une fonction intermédiaire pour calculer le développement de Taylor d'un réel x, qu'on va ensuite vectoriser
        reponse=0
        for i in range(len(derivatives)):
            reponse+=(derivatives[i]/factorial(i))*(x**i)
        return reponse
    taylor=np.vectorize(taylor_intermed)
    return taylor(X)

def taylor2(X, f, n):
    derivees=[f(0)]
    for k in range(n):
        f=grad(f)
        derivees.append(f(0.))
    return taylor1(X,derivees)

#4.2

def custom(X):
    return 2*np.sin(X)+np.cos(X/4)

def ln_decale(X):
    return np.log(1+X)




FONCTIONS={'sinus':np.sin,'cosinus':np.cos, 'exp':np.exp,"ln decalé":ln_decale,'custom':custom}
#La clef est le nom de la fonction à chaque fois

#4.3 

with st.sidebar : func=st.selectbox("Choisissez votre fonction",FONCTIONS)
with st.sidebar : n = st.select_slider("Choisissez l'ordre n de l'approximation", np.arange(0,11,1), value=3)
with st.sidebar : largeur = st.select_slider("Choisissez la largeur de l'intervalle étudié", np.arange(2,20.1,0.1), value=12.6)
#np.arange exclut la borne supérieure (ça fait de nombreux zéros après la virgule pour la borne 20, je n'ai pas réussi à régler ça)

#4.4

X=np.array(np.linspace(-largeur/2,largeur/2,400))
#Note à moi-même : arange est pratique quand on veut choisir l'espacement entre les valeurs, linspace quand on veut choisir le nombre d'échantillons
f=FONCTIONS[func]
Y=f(X)
Y_taylor=taylor2(X,f,n)

fig,ax=plt.subplots() #On crée un environnement d'affichage
ax.plot(X,Y)
ax.plot(X,Y_taylor)
st.pyplot(fig)

#Pour n ficé, la qualité de l'approximation diminue lorsqu'on s'éloigne de 0, car le développement en série entière peut être identifié au développement limité de la fonction en 0.
#Pour ln décalé, lorsqu'on augmente n, l'approximation "décroche" de la courbe initiale, car les polynômes croissent beaucoup plus vite que ln.