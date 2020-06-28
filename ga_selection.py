
#Generic Algorithm
from genetic import GeneticSelectionCV
#Linear Models
from sklearn import linear_model

def GASelection(feature_n, 
                        classes, 
                        cv=5,
                        scoring="accuracy",
                        max_features=3,
                        n_population=100, 
                        crossover_proba=0.5, 
                        mutation_proba=0.2,
                        n_generations=150,
                        crossover_independent_proba=0.5,
                        mutation_independent_proba=0.05,
                        tournament_size=3,
                        n_gen_no_change=10, 
                        caching=True):
    
    estimator = linear_model.LogisticRegression(solver="liblinear")
    linear_selector = GeneticSelectionCV(estimator,
                                cv=cv,
                                scoring=scoring,
                                max_features=max_features,
                                n_population=n_population, 
                                crossover_proba=crossover_proba, 
                                mutation_proba=mutation_proba,
                                n_generations=n_generations,
                                crossover_independent_proba=crossover_independent_proba,
                                mutation_independent_proba=mutation_independent_proba,
                                tournament_size=tournament_size,
                                n_gen_no_change=n_gen_no_change, 
                                caching=caching,
                                n_jobs=-1)
    linear_selector = linear_selector.fit(feature_n, classes)
    hist, shist, selfeat = linear_selector.analysis()
    return hist, selfeat, shist, linear_selector
