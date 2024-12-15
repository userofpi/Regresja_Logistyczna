# Removed erroneous import; LinearRegression is not used or relevant

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go


# dane

dane = pd.read_csv('Dane_praca_zad2.csv', sep=';')
print(dane['Liczba osob ktore znalazly prace'])

print(dane.head())

#Zbudować model logitowy dla prawdopodobieństwa znalezienia
# pracy w zależności od wieku rejestrującego się oraz jego stażu pracy

# zmienne obcjasniajace Staż i wiek bezrobotnych
# zmienna objasniana prawdopodobienstwo czyli Liczba bezrobotnych ktorzy znalezli prace/Liczba badanych

dane['prawdopodobienstwo'] = dane['Liczba osob ktore znalazly prace']/dane['Liczba badanych bezrobotnych']
#

print(dane.head())
X = dane[["Sredni staz pracy (lata)", "Wiek bezrobotnych(lata)"]]
# dodaj "Wiek bezrobotnych(lata)" do X



# podziel dane na uczące i testowe
X_treningowe, X_testowe, y_treningowe, y_testowe = train_test_split(X, dane['prawdopodobienstwo'], test_size=0.2, random_state=42)

print(y_treningowe)

print(X_treningowe.head())

X_treningowe=sm.add_constant(X_treningowe)

wynik= sm.Logit(y_treningowe, X_treningowe).fit()
print(wynik.summary())
X_testowe = sm.add_constant(X_testowe)


y_predykcje=wynik.predict(X_testowe)

print(y_predykcje)
print(y_testowe)


zdarzenie_przeciwne=dane['prawdopodobienstwo']/(1-dane['prawdopodobienstwo'])

# logarytm naturalny z zdarzenie_przeciwne
log= np.log(zdarzenie_przeciwne)
print(log)

# najmniesze kwadraty b0
b0=wynik.params[0]
print(b0)

# Tworzenie macierzy X dla dwóch zmiennych objaśniających
X = np.vstack([
    np.ones_like(dane["Sredni staz pracy (lata)"]),  # Kolumna jedynek (dla wyrazu wolnego)
    dane["Sredni staz pracy (lata)"],  # Pierwsza zmienna objaśniająca
    dane["Wiek bezrobotnych(lata)"]  # Druga zmienna objaśniająca
]).T

# Wzór: b = (X^T * X)^(-1) * X^T * log
b = np.linalg.inv(X.T @ X) @ X.T @ log

# Wyniki współczynników
b0, b1, b2 = b[0], b[1], b[2]
# fukcja exp
print(f"Współczynniki:")
print(f"b0 (wyraz wolny) = {b0}")
print(f"b1 (współczynnik zmiennej 1) = {b1}")
print(f"b2 (współczynnik zmiennej 2) = {b2}")
# X = dane[["Sredni staz pracy (lata)", "Wiek bezrobotnych(lata)"]]

# wykres  funckji
# os X srednia staz pracy i wiek bezrobotnego
# os Y prawdopodobiestwo
# Tworzymy siatkę wartości zmiennych niezależnych (staż pracy i wiek)
staz_pracy = np.linspace(dane['Sredni staz pracy (lata)'].min(), dane['Sredni staz pracy (lata)'].max(), 50)
wiek = np.linspace(dane['Wiek bezrobotnych(lata)'].min(), dane['Wiek bezrobotnych(lata)'].max(), 50)
staz_pracy, wiek = np.meshgrid(staz_pracy, wiek)

# Obliczenie prawdopodobieństwa z funkcji logistycznej
# logit = b0 + b1 * staz_pracy + b2 * wiek
# prawdopodobieństwo = 1 / (1 + exp(-logit))
logit = b0 + b1 * staz_pracy + b2 * wiek
prawdopodobienstwo = 1 / (1 + np.exp(-logit))

# Tworzenie wykresu 3D
fig = go.Figure(data=[go.Surface(z=prawdopodobienstwo, x=staz_pracy, y=wiek)])

# Ustawienia etykiet
fig.update_layout(
    title="Interaktywny wykres regresji logistycznej",
    scene=dict(
        xaxis_title="Średni staż pracy (lata)",
        yaxis_title="Wiek bezrobotnych (lata)",
        zaxis_title="Prawdopodobieństwo"
    )
)

# Wyświetlenie wykresu
fig.show()


#model Liniowy

Wynik_liniowy=sm.OLS(y_treningowe,X_treningowe).fit()
print(Wynik_liniowy.summary())

X_testowe = sm.add_constant(X_testowe)
y_predykcje=Wynik_liniowy.predict(X_testowe)

fig = go.Figure(data=[go.Surface(z=y_predykcje, x=staz_pracy, y=wiek)])

#Ustawienia etykiet
fig.update_layout(
    title="Interaktywny wykres regresji liniowej",
    scene=dict(
        xaxis_title="Średni staż pracy (lata)",
        yaxis_title="Wiek bezrobotnych (lata)",
        zaxis_title="Prawdopodobieństwo"
    )
)

# Wyświetlenie wykresu
fig.show()



