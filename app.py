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
    for k in range(1,n):
        f=grad(f)
        derivees.append(f(0.))
    return taylor1(X,derivees)

#4.2

def custom_unvectorized(X):
    return 2*np.sin(X)+np.cos(X/4)

custom=np.vectorize(custom_unvectorized)

def ln_decale_unvectorized(X):
    return np.ln(1+X)

ln_decale=np.vectorize(ln_decale_unvectorized)

FONCTIONS={np.sin:"sinus",np.cos:"cosinus",np.exp:"exp",ln_decale:"ln decalé",custom:"custom"}

st.selectbox("Choisissez votre fonction",FONCTIONS)