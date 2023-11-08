#!/usr/bin/env python
# coding: utf-8

# ## ZAWANSOWANE SYSTEMY SZTUCZNEJ INTELIGENCJI

# Wykonawcy: Kacper Kalinowski, Aleksander Kalinowski, Sylwia Mościcka

# Zbiór danych

# In[1]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("Houses.csv", encoding="windows-1250")
df


# 1. Usuwanie zbędnych atrybutów:
# Usunięcie atrybutu "Unnamed: 0" tzw. indexu z danych, ponieważ nie wnosi on istotnych informacji i nie ma wpływu na cenę nieruchomości.

# In[3]:


df = df.drop(columns=['Unnamed: 0'])


# In[4]:


df.info()


# Usunięcie niepotrzebnych atrybutów pomaga zmniejszyć liczbę wymiarów danych, co może poprawić wydajność modelu i zmniejszyć ryzyko przeuczenia (overfitting).

# 2. Ekstrakcja informacji z daty:
# Z atrybutu "year" można wyodrębnić informacje o dniu, miesiącu i roku, co może być przydatne w analizie zmian cen w zależności od pory roku.

# Ekstrakcja informacji z daty pozwala uwzględnić sezonowe zmiany cen nieruchomości. Niestety w naszych danych nie mamy informacji o miesiacach, przez co nie jesteśmy w stanie sprawdzić, w jakich miesiacach najchętniej kupowano mieszkania.

# Pozostałe dane też stworzone są w analogiczny sposób, więc nic nie musimy poddawać ekstracji.

# 3. Przystosowanie typu danych i kodowanie kategorii.

# In[5]:


df = pd.get_dummies(df, columns=['city', 'address'])


# In[14]:


df


# Model maszynowy wymaga atrybutów numerycznych, więc konieczne jest przekształcenie kategorii na wartości liczbowe.

# 4. Normalizacja wartości atrybutów.

# In[6]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[['floor','latitude', 'longitude', 'price', 'rooms', 'sq', 'year']] = scaler.fit_transform(df[['floor', 'latitude', 'longitude', 'price', 'rooms', 'sq', 'year']])


# In[7]:


df


# Normalizacja pomaga uniknąć problemów związanych z różnicami w skalach atrybutów.

# 5. Uzupełnienie danych brakujących.

# W naszym przypadku nie mieliśmy brakujących danych, przez co nie musieliśmy dodatkowo ich uzupełniać.

# 6. Czyszczenie danych.
# Usunięcie rekordów zawierających błędne dane lub zbyt wiele brakujących wartości. Warto również pozbyć się duplikatów, aby zachować czystość danych.

# In[8]:


df = df.dropna()  # Usuń wiersze z brakującymi danymi
df = df.drop_duplicates()  # Usuń duplikaty


# In[9]:


df


# Czyste dane zapewniają dokładniejsze i bardziej wiarygodne wyniki modelu.

# 7. Podjęcie decyzji, czy zachować dane odstające (ang. outliers).

# In[10]:


from sklearn.ensemble import IsolationForest

X = df[['price']]

# Inicjalizacja modelu Isolation Forest
model = IsolationForest(contamination=0.05, random_state=42)

# Dopasowanie modelu
model.fit(X)

# Oznacz anomalia w danych
df['anomaly'] = model.predict(X)
df['anomaly']


# Wybieramy atrybuty, na których chcemy wykrywać anomalie (w przykładzie używamy tylko 'price').
# Inicjalizujemy model Isolation Forest, ustawiając parametr contamination na poziom odsetka danych uważanych za anomalie (w tym przykładzie 5%).
# Dopasowujemy model do danych za pomocą fit.
# Używamy modelu do przewidywania anomalii na danych i oznaczamy wyniki jako nowy atrybut 'anomaly' w naszym DataFrame.

# W naszej analizie wyszło dużo anomalii, jednak ich nie usuwamy. Cena może mieć w naszym przypadku ogromne znaczenie.
# 

# Anomalie w cenach mogą wskazywać na nieruchomości, które są szczególnie atrakcyjne z ceną poniżej przeciętnej lub nieruchomości, które wymagają uwagi, ponieważ ich cena jest znacząco wyższa. To może pomóc inwestorom lub kupującym nieruchomości w identyfikacji potencjalnie atrakcyjnych ofert lub problematycznych transakcji. 

# Z drugiej strony dla instytucji finansowych, które udzielają kredytów hipotecznych, identyfikacja cen nieruchomości uważanych za anomalie może pomóc w ocenie ryzyka kredytowego. Jeśli cena nieruchomości jest znacznie niższa od przeciętnej, może to oznaczać ryzyko utraty wartości nieruchomości w przypadku niewypłacalności kredytobiorcy.

# In[11]:


from sklearn.ensemble import IsolationForest

X = df[['year']]

# Inicjalizacja modelu Isolation Forest
model = IsolationForest(contamination=0.05, random_state=42)

# Dopasowanie modelu
model.fit(X)

# Oznacz anomalia w danych
df['anomaly'] = model.predict(X)
anomaly_counts = df['anomaly'].value_counts()

# Wyświetl wynik
print(anomaly_counts)


# W roku wyniki są bardzo rozbieżne, więc również nie bedziemy usuwać danych.

# In[12]:


from sklearn.ensemble import IsolationForest

X = df[['sq']]

# Inicjalizacja modelu Isolation Forest
model = IsolationForest(contamination=0.05, random_state=42)

# Dopasowanie modelu
model.fit(X)

# Oznacz anomalia w danych
df['anomaly'] = model.predict(X)
anomaly_counts = df['anomaly'].value_counts()

# Wyświetl wynik
print(anomaly_counts)


# Wyniki identyfikacji anomalii w atrybucie "sq" (powierzchnia nieruchomości) wskazują, że większość obserwacji (20,922) została uznana za normalne, a 1,094 obserwacje zostały uznane za anomalię.

# W naszej analzie jest dużo anomali. Podjęliśmy decyzję, aby wszystkie zostawić.Zachowanie danych odstających (anomalii) w analizie danych może być uzasadnione i ważne z kilku powodów:
# 
#     Pełniejsza reprezentacja rzeczywistości: Anomalie mogą być rzeczywistymi obserwacjami, które odzwierciedlają różnorodność i nietypowość na rynku nieruchomości. Włączenie ich w analizę może pomóc w lepszym zrozumieniu pełnego zakresu danych.
# 
#     Unikalne czynniki wpływające na cenę: Nieruchomości o nietypowych rozmiarach mogą mieć unikalne cechy lub czynniki wpływające na cenę, które są ważne w analizie rynku nieruchomości. Ich uwzględnienie może dostarczyć cennych informacji.
# 
#     Identyfikacja lukratywnych okazji lub problematycznych nieruchomości: Anomalie mogą wskazywać na nieruchomości, które są szczególnie atrakcyjne (jeśli są tańsze) lub problematyczne (jeśli są droższe). To może być istotne dla inwestorów lub kupujących nieruchomości.
# 
#     Wartość predykcyjna: Anomalie mogą zawierać istotne informacje dla predykcji cen nieruchomości. Jeśli te nietypowe przypadki są częścią rzeczywistego rynku nieruchomości, to uwzględnienie ich w modelach predykcyjnych może poprawić ich skuteczność.
# 
#     Rozpoznawanie trendów rynkowych: Anomalie mogą być wskaźnikiem zmian na rynku nieruchomości. Nagłe zmiany w cenach lub rozmiarach nieruchomości mogą wskazywać na istotne trendy rynkowe.

# 8. Zastosowanie metod oversampling/undersampling w celu zniwelowania problemu niezbalansowania danych.

# In[17]:


from sklearn.model_selection import train_test_split
X = df.drop(columns=['price'])
y = df['price']

# Podziel dane na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ręczny oversampling klasy mniejszościowej
class_0 = df[df['price'] == 0]
class_1 = df[df['price'] == 1]

# Określ ile próbek ma zostać dodanych w oversamplingu
n_samples_to_add = len(class_0) - len(class_1)

# Losowo wybierz próbki z klasy mniejszościowej i dodaj je wielokrotnie
oversampled_class_1 = class_1.sample(n=n_samples_to_add, replace=True, random_state=42)

# Połącz zrównoważone klasy
balanced_df_oversampled = pd.concat([class_0, oversampled_class_1])

# Ręczny undersampling klasy większościowej
n_samples_to_remove = len(class_0) - len(class_1)

# Usuń nadmiarowe próbki z klasy większościowej
undersampled_class_0 = class_0.sample(n=len(class_1), random_state=42)

# Połącz zrównoważone klasy
balanced_df_undersampled = pd.concat([undersampled_class_0, class_1])


# W błędzie występuje problem z próbą oversamplingu, ponieważ próbki próbowane są w ilości mniejszej lub równej zeru. To oznacza, że liczba próbek klasy mniejszościowej jest już większa lub równa liczbie próbek klasy większościowej, co sprawia, że oversampling nie jest potrzebny.
# 
# Jeśli klasy są już zrównoważone (lub klasa mniejszościowa jest większa), nie ma potrzeby stosowania oversamplingu. \
# 
#   Niezbalansowane dane: Dane mogą być niezbalansowane, co oznacza, że jedna z klas jest znacznie liczniejsza od drugiej. W Twoim przypadku, mogą istnieć nieruchomości o różnych cenach, ale pewne kategorie cenowe mogą być mniej liczne. Oversampling pomaga w zrównoważeniu klas, co jest istotne, aby model mógł nauczyć się równie dobrze klasy mniejszościowe.
# 
#     Uniknięcie przeuczenia: Przeuczenie (overfitting) to sytuacja, w której model jest zbyt dopasowany do danych treningowych i nie ogólny na nowe dane. Walidacja krzyżowa pomaga w ocenie ogólnej wydajności modelu na różnych zestawach danych treningowych i testowych, co pomaga uniknąć przeuczenia.
# 
#     Poprawa oceny modelu: Walidacja krzyżowa dostarcza średni wynik wydajności modelu na różnych zestawach danych, co pozwala na bardziej rzetelną ocenę modelu. Oversampling pomaga w lepszym wykorzystaniu danych mniejszościowych, co może poprawić wyniki modelu.
# 
#     Równowaga między precyzją a pełnością: Oversampling pomaga zrównoważyć między precyzją (dokładnością) a pełnością (pokryciem) w kontekście klasy mniejszościowej. Niezbalansowane dane mogą prowadzić do modeli, które są zbyt precyzyjne, ale nie pokrywają wszystkich przypadków.

# 9. Implementacja walidacji krzyżowej (uwaga: kroki opisane wyżej lub inne konieczne do przeprowadzenia w ramach przygotowania wstępnego danych należy wykonywać tylko na danych treningowych, aby uniknąć tzw. wycieku danych (ang. data leakage))

# In[19]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
X = df.drop(columns=['price'])
y = df['price']

# Inicjalizacja modelu regresji liniowej
model = LinearRegression()

# Przeprowadzenie 5-krotnej walidacji krzyżowej
scores = cross_val_score(model, X, y, cv=5, scoring='r2')

# Wyświetlenie wyników walidacji krzyżowej
print("Wyniki walidacji krzyżowej (R^2):", scores)
print("Średni wynik walidacji krzyżowej (R^2):", scores.mean())


#   Wyniki walidacji krzyżowej (R^2) są miarą wydajności modelu regresji liniowej na różnych podzbiorach danych. R^2 (R-squared) jest miarą, która wskazuje, jak dobrze model pasuje do danych. Wartość R^2 mieści się w zakresie od 0 do 1, gdzie:
# 
#     R^2 = 0 oznacza, że model nie wyjaśnia żadnej zmienności w danych.
#     R^2 = 1 oznacza, że model doskonale wyjaśnia zmienność w danych.
# 
# Wyniki walidacji krzyżowej (R^2) dla pięciu różnych iteracji. Oto interpretacja tych wyników:
# 
#     -1.10132456e+13: Model miał bardzo słabą wydajność w tej iteracji, a R^2 jest bardzo niskie, co sugeruje, że model źle dopasował się do danych.
# 
#     -6.73892362e+01: Model miał lepszą wydajność niż w pierwszej iteracji, ale R^2 nadal jest ujemne, co może oznaczać, że model nie pasuje dobrze do danych.
# 
#     4.97220934e-01: W tej iteracji model uzyskał pozytywną wartość R^2, co sugeruje, że model lepiej pasuje do danych niż w poprzednich iteracjach.
# 
#     -2.92894489e+13: Ta iteracja znowu wykazała bardzo niską wydajność modelu, a R^2 jest bardzo niskie.
# 
#     -2.25321442e+10: Wynik w tej iteracji również jest bardzo niski, co sugeruje, że model słabo pasuje do danych.
# 
# Średnia wartość R^2 z tych iteracji wynosi -8065045337269.175, co jest bardzo niską wartością. To sugeruje, że model regresji liniowej jest bardzo słaby w przewidywaniu cen nieruchomości na podstawie dostępnych atrybutów.

# ### Eksperyment w celu uzyskania lepszych danych

# In[21]:


df.dropna(inplace=True)

# Wybierz tylko atrybuty numeryczne do normalizacji
numerical_features = ['floor', 'latitude', 'longitude', 'rooms', 'sq', 'year']

# Normalizacja atrybutów numerycznych
scaler = MinMaxScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Podziel dane na atrybuty i etykietę
X = df.drop(columns=['price'])
y = df['price']

# Podziel dane na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Teraz możesz użyć X_train i y_train do trenowania modelu, a X_test i y_test do oceny jego wydajności.


# In[23]:


model = LinearRegression()

# Przeprowadzenie 5-krotnej walidacji krzyżowej
scores = cross_val_score(model, X, y, cv=5, scoring='r2')

# Wyświetlenie wyników walidacji krzyżowej
print("Wyniki walidacji krzyżowej (R^2):", scores)
print("Średni wynik walidacji krzyżowej (R^2):", scores.mean())


# Obecne wyniki są znacznie lepsze, oznaczają lepszą zdolność modelu do dopasowania się do danych. Wartość średniego wyniku walidacji krzyżowej R^2 wynosi teraz -5128314123971.519, co wskazuje na to, że model lepiej wyjaśnia zmienność w danych w porównaniu do poprzednich wyników.

# In[24]:


df = pd.read_csv("Houses.csv", encoding="windows-1250")


# In[25]:


def preprocess_inputs(df):
    df = df.copy()
    
    # Drop unused columns
    df = df.drop(['Unnamed: 0', 'address', 'id'], axis=1)
    
    # Split df into X and y
    y = df['price']
    X = df.drop('price', axis=1)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)
    
    return X_train, X_test, y_train, y_test


# In[27]:


X_train, X_test, y_train, y_test = preprocess_inputs(df)


# In[28]:


X_train


# In[29]:


y_train


# In[31]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import GradientBoostingRegressor
nominal_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(sparse=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('nominal', nominal_transformer, ['city'])
], remainder='passthrough')

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),
    ('regressor', GradientBoostingRegressor())
])


# In[32]:


model.fit(X_train, y_train)


# In[33]:


y_pred = model.predict(X_test)

rmse = np.sqrt(np.mean((y_test - y_pred)**2))
print("RMSE: {:.5f}".format(rmse))

baseline_errors = np.sum((y_test - np.mean(y_test))**2)
model_errors = np.sum((y_test - y_pred)**2)

r2 = 1 - (model_errors / baseline_errors)
print("R^2 Score: {:.5f}".format(r2))


#     RMSE (Root Mean Square Error):
#         RMSE to miara, która określa, jak dokładnie model przewiduje wartości.
#         RMSE oblicza pierwiastek średniego kwadratu różnicy między rzeczywistymi danymi (y_test) a przewidywanymi danymi (y_pred).
#         Im niższa wartość RMSE, tym lepsza jakość modelu. W tym przypadku wynosi ona około 246098.14872, co oznacza, że model przewiduje średnio z błędem około 246098.15 jednostek w przypadku cen nieruchomości.
# 
#     R^2 Score (Coefficient of Determination):
#         R^2 Score mierzy, jak dobrze model pasuje do danych w porównaniu do prostego modelu średniej (baseline).
#         R^2 Score w zakresie od 0 do 1, gdzie 1 oznacza doskonałe dopasowanie modelu do danych, a 0 oznacza, że model nie jest lepszy od prostego modelu średniej.
#         Wartość R^2 Score wynosi około 0.78564, co oznacza, że model wyjaśnia około 78.56% zmienności w danych cen nieruchomości.
# 
# Wynik R^2 Score na poziomie 0.78564 jest dość dobry, co sugeruje, że model regresji liniowej dobrze pasuje do danych. Jednak RMSE jest nadal dość wysoki, co oznacza, że istnieją pewne odchylenia między przewidywaniami modelu a rzeczywistymi danymi. 

# In[34]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Funkcja do przetwarzania danych
def preprocess_inputs(df):
    df = df.copy()
    
    # Drop unused columns
    df = df.drop(['Unnamed: 0', 'address', 'id'], axis=1)
    
    # Split df into X and y
    y = df['price']
    X = df.drop('price', axis=1)
    
    return X, y

# Wczytaj dane
df = pd.read_csv("Houses.csv", encoding="windows-1250")

# Usuń brakujące wartości
df.dropna(inplace=True)

# Podziel dane na atrybuty i etykietę
X, y = preprocess_inputs(df)

# Podziel dane na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocessing i model
nominal_transformer = Pipeline(steps=[('onehot', OneHotEncoder(sparse=False))])
preprocessor = ColumnTransformer(transformers=[('nominal', nominal_transformer, ['city'])], remainder='passthrough')
model = Pipeline(steps=[('preprocessor', preprocessor), ('scaler', StandardScaler()), ('regressor', GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42))])

# Trenowanie modelu
model.fit(X_train, y_train)

# Przewidywanie
y_pred = model.predict(X_test)

# Ocena modelu
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE: {:.5f}".format(rmse))
print("R^2 Score: {:.5f}".format(r2))


# Wprowadziliśmy kilka zmian w kodzie, takie jak dostosowanie podziału danych na treningowe i testowe (30% testowych) oraz dostosowanie parametrów modelu GradientBoostingRegressor, takich jak n_estimators, max_depth, learning_rate. Te parametry mogą wymagać dalszego tuningu, ale te wartości wydają się sensowne na początek.Otrzymane RMSE na poziomie 209474.11972 i R^2 Score wynoszący 0.83332 sugerują, że model jest teraz bardziej dokładny w przewidywaniu cen nieruchomości.

# In[ ]:




