## Question 1: How many examples are there in the dataset?

(2494, 28)

## Question 2: What is the distribution of the target?
Census_Region
East North Central    758
West North Central    358
Middle Atlantic       334
South Atlantic        248
Pacific               243
Mountain              190
West South Central    172
East South Central     97
New England            94
Name: count, dtype: int64


## Question 3: What are the features that can be used to predict the target?


Index(['RespondentID',
       'What_would_you_call_the_part_of_the_country_you_live_in_now',
       'How_much_do_you_personally_identify_as_a_Midwesterner',
       'Do_you_consider_Illinois_state_as_part_of_the_Midwest',
       'Do_you_consider_Indiana_state_as_part_of_the_Midwest',
       'Do_you_consider_Iowa_state_as_part_of_the_Midwest',
       'Do_you_consider_Kansas_state_as_part_of_the_Midwest',
       'Do_you_consider_Michigan_state_as_part_of_the_Midwest',
       'Do_you_consider_Minnesota_state_as_part_of_the_Midwest',
       'Do_you_consider_Missouri_state_as_part_of_the_Midwest',
       'Do_you_consider_Nebraska_state_as_part_of_the_Midwest',
       'Do_you_consider_North_Dakota_state_as_part_of_the_Midwest',
       'Do_you_consider_Ohio_state_as_part_of_the_Midwest',
       'Do_you_consider_South_Dakota_state_as_part_of_the_Midwest',
       'Do_you_consider_Wisconsin_state_as_part_of_the_Midwest',
       'Do_you_consider_Arkansas_state_as_part_of_the_Midwest',
       'Do_you_consider_Colorado_state_as_part_of_the_Midwest',
       'Do_you_consider_Kentucky_state_as_part_of_the_Midwest',
       'Do_you_consider_Oklahoma_state_as_part_of_the_Midwest',
       'Do_you_consider_Pennsylvania_state_as_part_of_the_Midwest',
       'Do_you_consider_West_Virginia_state_as_part_of_the_Midwest',
       'Do_you_consider_Montana_state_as_part_of_the_Midwest',
       'Do_you_consider_Wyoming_state_as_part_of_the_Midwest', 'Gender', 'Age',
       'Household_Income', 'Education',
       'In_what_ZIP_code_is_your_home_located'],
      dtype='str')



## Question 4: Are there any missing values in the dataset?


RespondentID                                                   0
What_would_you_call_the_part_of_the_country_you_live_in_now    0
How_much_do_you_personally_identify_as_a_Midwesterner          0
Do_you_consider_Illinois_state_as_part_of_the_Midwest          0
Do_you_consider_Indiana_state_as_part_of_the_Midwest           0
Do_you_consider_Iowa_state_as_part_of_the_Midwest              0
Do_you_consider_Kansas_state_as_part_of_the_Midwest            0
Do_you_consider_Michigan_state_as_part_of_the_Midwest          0
Do_you_consider_Minnesota_state_as_part_of_the_Midwest         0
Do_you_consider_Missouri_state_as_part_of_the_Midwest          0
Do_you_consider_Nebraska_state_as_part_of_the_Midwest          0
Do_you_consider_North_Dakota_state_as_part_of_the_Midwest      0
Do_you_consider_Ohio_state_as_part_of_the_Midwest              0
Do_you_consider_South_Dakota_state_as_part_of_the_Midwest      0
Do_you_consider_Wisconsin_state_as_part_of_the_Midwest         0
Do_you_consider_Arkansas_state_as_part_of_the_Midwest          0
Do_you_consider_Colorado_state_as_part_of_the_Midwest          0
Do_you_consider_Kentucky_state_as_part_of_the_Midwest          0
Do_you_consider_Oklahoma_state_as_part_of_the_Midwest          0
Do_you_consider_Pennsylvania_state_as_part_of_the_Midwest      0
Do_you_consider_West_Virginia_state_as_part_of_the_Midwest     0
Do_you_consider_Montana_state_as_part_of_the_Midwest           0
Do_you_consider_Wyoming_state_as_part_of_the_Midwest           0
Gender                                                         0
Age                                                            0
Household_Income                                               0
Education                                                      0
In_what_ZIP_code_is_your_home_located                          0
dtype: int64



## Question 5: What is the most common answer to "How much do you personally identify as a Midwesterner"?

How_much_do_you_personally_identify_as_a_Midwesterner
Not at all    965
A lot         697
Some          528
Not much      304
Name: count, dtype: int64


## Bonus: Explore another feature

Gender
Female    1326
Male      1168
Name: count, dtype: int64

Gender
Female    0.531676
Male      0.468324
Name: proportion, dtype: float64

Age
45-60    705
30-44    634
> 60     634
18-29    521
Name: count, dtype: int64

6. Among the three models, which one has the best recall :

Logistic Regression:
               precision    recall  f1-score   support

        other       0.00      0.00      0.00       669
North Central       0.55      1.00      0.71       825

     accuracy                           0.55      1494
    macro avg       0.28      0.50      0.36      1494
 weighted avg       0.30      0.55      0.39      1494

Random Forest:
               precision    recall  f1-score   support

        other       0.95      0.92      0.94       669
North Central       0.94      0.96      0.95       825

     accuracy                           0.94      1494
    macro avg       0.94      0.94      0.94      1494
 weighted avg       0.94      0.94      0.94      1494

Gradient Boosting:
               precision    recall  f1-score   support

        other       0.96      0.99      0.97       669
North Central       0.99      0.97      0.98       825

     accuracy                           0.98      1494
    macro avg       0.98      0.98      0.98      1494
 weighted avg       0.98      0.98      0.98      1494

Recall pour la classe positive (« North Central ») :

    Logistic Regression : 0.0000
    Random Forest : 0.9238
    Gradient Boosting : 0.9895

Le modèle ayant le meilleur recall est : Gradient Boosting Il identifie presque tous les vrais cas « North Central ».

7 Quel modèle a la meilleure application pratique ?
Structure de coûts : 
•	Faux positif (FP) = -10
•	Faux négatif (FN) = -1
•	Vrai positif (TP) = +5
•	Vrai négatif (TN) = +2
Scores pratiques : 
•	Logistic Regression : 981
•	Random Forest : 4293
•	Gradient Boosting : 4629
Le modèle le plus pertinent en pratique est : Gradient Boosting Il obtient le score pratique le plus élevé grâce à : 
•	Un recall très élevé
•	Une très bonne précision
•	Un excellent compromis global





8 Quel modèle généralise le mieux ?
Résultats de la validation croisée :

| Modèle | Accuracy train | Accuracy test | Écart |
|--------|---------------|--------------|--------|
| Logistic Regression | 0.5525 | 0.5525 | 0.0000 |
| Random Forest | 1.0000 | 0.9274 | 0.0726 |
| Gradient Boosting | 1.0000 | 0.9479 | 0.0521 |

•	Plus petit écart train/test : Logistic Regression
•	Plus grand écart (surapprentissage) : Random Forest
Techniquement, le modèle qui généralise le mieux (écart le plus faible) est : Logistic Regression Cependant, ses performances globales sont faibles. --- 
Choix final pour une application réelle
Si l’on devait choisir un modèle pour une application réelle : Gradient Boosting Raisons : 
•	Meilleur recall
•	Meilleur score pratique
•	Excellentes performances globales
•	Écart train/test raisonnable
