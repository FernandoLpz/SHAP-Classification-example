import optuna
import numpy as np

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier


class Classifier:

    def __init__(self, **kwargs) -> None:
        self.x_train = kwargs['x_train']
        self.x_test = kwargs['x_test']
        self.y_train = kwargs['y_train']
        self.y_test = kwargs['y_test']

        self._train()

        pass

    
    def _optimize(self, trial) -> np.ndarray:
    
        # Definition of space search
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
        n_estimators = trial.suggest_int('n_estimators', 10, 500)
        max_depth = trial.suggest_int('max_depth', 2, 10)

        # Classifier definition
        model = RandomForestClassifier(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    criterion=criterion, 
                                    n_jobs=-1)

        avg_accuracy = []

        # Definition of k-fold cross validation
        k_fold = KFold(n_splits=3)

        for train_idx, test_idx in k_fold.split(self.x_train.values, self.y_train.values):
            
            # Training fold
            x_train = self.x_train.values[train_idx]
            y_train = self.y_train.values[train_idx]
            
            # Testing fold
            x_test = self.x_train.values[test_idx]
            y_test = self.y_train.values[test_idx]

            # Training
            model.fit(x_train, y_train)

            # Save accuracy
            avg_accuracy.append(model.score(x_test, y_test))
        
        return np.mean(avg_accuracy)


    def _train(self) -> None:
        
        # Hyperparameter Optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(self._optimize, n_trials=100)

        # Train with optimal paramters
        self.model = RandomForestClassifier(n_estimators=study.best_params['n_estimators'],
                                            criterion=study.best_params['criterion'],
                                            max_depth=study.best_params['max_depth'])
        
        self.model.fit(self.x_train, self.y_train)