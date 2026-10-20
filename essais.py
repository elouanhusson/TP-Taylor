import streamlit as st

#1.3)
valeur = st.slider("Choisissez une valeur", 0, 10)
st.write("Vous avez choisi :", valeur)

#Le script est exécuté plusieurs fois, car il met à jour la valeur de "Vous avez choisi" en continu.

#2

import autograd.numpy as np
from autograd import grad

#2.1

X=np.linspace(0,2*np.pi,5)

#2.2 et Q3

sin_der=np.vectorize(grad(np.sin))
print(sin_der(X))
print(np.isclose(sin_der(X),np.cos(X),10**(-5))) #Les écarts sont pratiquement nuls, c'est validé

#3

from math import factorial
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg') #Je n'arrivais pas à afficher mes graphiques, j'ai trouvé sur Internet qu'il fallait faire ça

#3.1 

def taylor1(X, derivatives):
    def taylor_intermed(x): #On crée une fonction intermédiaire pour calculer le développement de Taylor d'un réel x, qu'on va ensuite vectoriser
        reponse=0
        for i in range(len(derivatives)):
            reponse+=(derivatives[i]/factorial(i))*(x**i)
        return reponse
    taylor=np.vectorize(taylor_intermed)
    return taylor(X)

sinus10 = [0, 1, 0, -1, 0, 1, 0, -1, 0, 1]
X = np.linspace(-2*np.pi, 2*np.pi, 400)

plt.figure(1)
plt.plot(X,taylor1(X,sinus10))
plt.plot(X,np.sin(X))
plt.show()

#3.2

def taylor2(X, f, n):
    derivees=[f(0)]
    for k in range(1,n):
        f=grad(f)
        derivees.append(f(0.))
    return taylor1(X,derivees)

# sinus au degré 10, à comparer avec np.sin(X)
Y = np.sin(X)
Y2 = taylor2(X, np.sin, 10)

#1

plt.figure(2)
plt.plot(X,Y)
plt.plot(X,Y2)
plt.show()

#2

close_enough=np.isclose(Y,Y2,0.001) #Nous indique si l'approximation est proche à 10^(-3) ou non de la valeur réelle en renvoyant un tableau de booléens pour chaque valeur du domaine
print(100*list(close_enough).count(True)/len(close_enough)) #Compte le ratio de "True" sur le nombre de valeurs du domaine

#3

plt.figure(3)
plt.plot(X,np.cos(X))
plt.plot(X,taylor2(X,np.cos,0))
plt.show()

#L'approximation de degré 0 revient à localement approximer la fonction par une fonction constante. 
#Ici, comme cos atteint un maximum en 0, il s'avère que c'est également la tangente à la courbe de cos en 0.

#3.3

def custom(X):
    return 2*np.sin(X)+np.cos(X/4)

Y3=custom(X)
Y4=taylor2(X,custom,14)

plt.figure(4)
plt.plot(X,Y3)
plt.plot(X,Y4)
plt.show()

close=np.isclose(Y3,Y4,0.001) #Même technique qu'avant
print(100*list(close).count(True)/len(close))

#On a calculé ici l'approximation au degré 14 contre au degré 10 pour fig3. 
#Conséquemment, le ratio de domaine de coïncidence à 10^(-3) est plus élevé (59,5% contre 40,0%).

#4



        




