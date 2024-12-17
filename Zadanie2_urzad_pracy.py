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

from scipy.stats import pearsonr

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

dane['logity']=dane['prawdopodobienstwo']/(1-dane['prawdopodobienstwo'])
dane['logity']=np.log(dane['logity'])
print(dane['logity'])
dane['logity_2']=1/(1+np.exp(-dane['logity']))

# podziel dane na uczące i testowe
X_treningowe, X_testowe, y_treningowe, y_testowe = train_test_split(X, dane['logity'], test_size=0.2, random_state=42)

y_treningowe_exp=1/(1+np.exp(-y_treningowe))
y_testowe_exp=1/(1+np.exp(-y_testowe))

print(y_treningowe)

print(y_treningowe_exp.head())

X_treningowe=sm.add_constant(X_treningowe)



wynik= sm.Logit(y_treningowe_exp, X_treningowe).fit()
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
log = b0 + b1 * staz_pracy + b2 * wiek
prawdopodobienstwo = 1 / (1 + np.exp(-log))

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

# correlations

corr, _ = pearsonr(dane['Liczba osob ktore znalazly prace'], dane['Liczba badanych bezrobotnych'])
print("Korelacja zmiennych objasniających):\n", corr)

# wypisz wszystkie korelacje po koleji
for i in dane.columns:
    corr, _ = pearsonr(dane[i], dane['logity'])
    # nazwa kolumny i corr
    print(i, corr)

# widzimy że mniej skorelowany z prawdopodobienstwem jest Wiek oraz silną korelację zmiennych objasniajacych
# dlatego tworzymy nowy model z jedną zmienna wiek
X_model_2 = X_treningowe[["Wiek bezrobotnych(lata)"]]
X_model_2 = sm.add_constant(X_model_2)
wynik_2 = sm.Logit(y_treningowe_exp, X_model_2).fit()
print(wynik_2.summary())

X_testowe_wiek = sm.add_constant(X_testowe[["Wiek bezrobotnych(lata)"]])
y_predykcje_2 = wynik_2.predict(X_testowe_wiek)

print(y_predykcje_2)

b0, b1 = wynik_2.params[0], wynik_2.params[1]
# wzor funkcji sigmoid
wynik_funkcji = 1 / (1 + np.exp(-b0 - b1 * dane['Wiek bezrobotnych(lata)']))

# regresja logistyczna

plt.plot(dane['Wiek bezrobotnych(lata)'], wynik_funkcji, label="Regresja logistyczna")
# Scatter dla rzeczywistych wartości
plt.scatter(dane['Wiek bezrobotnych(lata)'], dane['logity_2'], label="Rzeczywiste dane", color='orange',
            alpha=0.7)

# Oznaczenia wykresu
plt.title("logity znalezienia pracy w zależności od wieku")
plt.xlabel("Wiek bezrobotnych (lata)")
plt.ylabel("logity")
# Ustawienie zakresu osi Y od 0 do 1
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.show()

# model liniowy z jedną zmienna objasiajaca
X_model_3=sm.add_constant(dane["Wiek bezrobotnych(lata)"])

wynik_3 = sm.OLS(dane['logity'], X_model_3).fit()
print(wynik_3.summary())
plt.scatter(dane['Wiek bezrobotnych(lata)'], dane['logity'], label="Rzeczywiste dane", color='orange')
plt.plot(dane[["Wiek bezrobotnych(lata)"]], wynik_3.predict(X_model_3), label="Regresja liniowa", color="red")
plt.title("Regresja liniowa zmiennej objasniajaca")
plt.xlabel("Wiek bezrobotnych (lata)")
plt.ylabel("logity")
plt.legend()
plt.grid(True)
plt.show()

# wykres dwoch zmiennych objasniajacych liniowy
X_treningowe_2_lin = sm.add_constant(X_treningowe)
wynik_lin = sm.OLS(y_treningowe, X_treningowe_2_lin).fit()
print(wynik_lin.summary())



# Tworzenie siatki wartości (staż pracy i wiek)
staz_pracy = np.linspace(dane['Sredni staz pracy (lata)'].min(), dane['Sredni staz pracy (lata)'].max(), 50)
wiek = np.linspace(dane['Wiek bezrobotnych(lata)'].min(), dane['Wiek bezrobotnych(lata)'].max(), 50)
staz_pracy, wiek = np.meshgrid(staz_pracy, wiek)

# Wyciągnięcie współczynników z regresji liniowej
b0, b1, b2 = wynik_lin.params[0], wynik_lin.params[1], wynik_lin.params[2]

# Obliczenie wartości funkcji liniowej
z = b0 + b1 * staz_pracy + b2 * wiek  # Płaszczyzna funkcji liniowej

# 1. Wykres 2D z matplotlib – funkcja liniowa tylko względem jednej zmiennej (np. wiek)
plt.plot(dane['Wiek bezrobotnych(lata)'], b0 + b2 * dane['Wiek bezrobotnych(lata)'], label="Regresja liniowa (2D)")
plt.title("Funkcja liniowa (2D) zależność od wieku")
plt.xlabel("Wiek bezrobotnych (lata)")
plt.ylabel("Wartość funkcji liniowej")
plt.ylim(0, 1)  # Ograniczenie osi Y opcjonalne
plt.legend()
plt.grid(True)
plt.show()

# 2. Wykres 3D z plotly – funkcja liniowa (płaszczyzna regresji)
fig = go.Figure(data=[go.Surface(z=z, x=staz_pracy, y=wiek)])

# Ustawienia wykresu
fig.update_layout(
    title="Płaszczyzna regresji liniowej w 3D",
    scene=dict(
        xaxis_title="Średni staż pracy (lata)",
        yaxis_title="Wiek bezrobotnych (lata)",
        zaxis_title="Prawodobodobieństwo znaleznienia pracy",
        zaxis=dict(range=[z.min(), z.max()])  # Skalowanie Z bazowe na wartościach regresji
    )
)
fig.show()
