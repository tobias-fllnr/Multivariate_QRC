import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

class Prediction:
    def __init__(self, observations: np.ndarray, data: np.ndarray, washout: int = 1000, train_length: int = 6000, test_length: int = 4000, model: str = "linear", ridge_alpha: float = 1.0):
        """Initializes the Prediction class.
        Args:
            observations (np.ndarray): The reservoir states or features.
            data (np.ndarray): The target data to predict.
            washout (int): Number of initial samples to discard.
            train_length (int): Length of the training dataset.
            test_length (int): Length of the testing dataset.
            model (str): Type of regression model ("linear" or "ridge").
            ridge_alpha (float): Regularization strength for Ridge regression.
        """
        self.observations = observations
        self.data = data
        self.washout = washout
        self.train_length = train_length
        self.test_length = test_length  
        self.model = model
        self.ridge_alpha = ridge_alpha

        if len(self.observations) != len(self.data):
            raise ValueError(f"observations and data must have the same length. Got {len(self.observations)} and {len(self.data)}")

        if len(self.observations) != self.washout + self.train_length + self.test_length:
            raise ValueError(f"observations length must equal washout + train_length + test_length. Got {len(self.observations)}, expected {self.washout + self.train_length + self.test_length}")

    def prediction_multi_step(self, max_steps: int=10) -> dict:
        """Performs multi-step ahead prediction.
        Args:
            max_steps (int): Maximum number of steps to predict ahead.
        Returns:
            dict: A dictionary containing RMSE results for each prediction step.
        """
        results_dict = {
            "rmse_train_average": np.zeros(max_steps),
            "rmse_test_average": np.zeros(max_steps),
            "rmse_train_list": np.zeros((max_steps, self.data.shape[1])),
            "rmse_test_list": np.zeros((max_steps, self.data.shape[1])),
            "nrmse_train_average": np.zeros(max_steps),
            "nrmse_test_average": np.zeros(max_steps),
            "nrmse_train_list": np.zeros((max_steps, self.data.shape[1])),
            "nrmse_test_list": np.zeros((max_steps, self.data.shape[1]))
        }
        for i in range(1, max_steps+1):
            targets = np.roll(self.data, -i, axis=0)
            targets = targets[self.washout:self.washout+self.train_length+self.test_length-i, :]
            observations = self.observations[self.washout:self.washout+self.train_length+self.test_length-i, :]
            rmse_train_average, rmse_test_average, rmse_train_list, rmse_test_list, nrmse_train_average, nrmse_test_average, nrmse_train_list, nrmse_test_list = self._trainer(observations, targets)
            results_dict["rmse_train_average"][i-1] = rmse_train_average
            results_dict["rmse_test_average"][i-1] = rmse_test_average
            results_dict["rmse_train_list"][i-1, :] = rmse_train_list
            results_dict["rmse_test_list"][i-1, :] = rmse_test_list
            results_dict["nrmse_train_average"][i-1] = nrmse_train_average
            results_dict["nrmse_test_average"][i-1] = nrmse_test_average
            results_dict["nrmse_train_list"][i-1, :] = nrmse_train_list
            results_dict["nrmse_test_list"][i-1, :] = nrmse_test_list
        return results_dict

    def _trainer(self, observations, targets):
        X_train, X_test, y_train, y_test = train_test_split(observations, targets, train_size=self.train_length, shuffle=False)

        if self.model == "linear":
            model = LinearRegression()
        elif self.model == "ridge":
            model = Ridge(alpha=self.ridge_alpha)
        else:
            raise ValueError(f"Unknown model: {self.model}")

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        rmse_train_list = root_mean_squared_error(y_train, y_train_pred, multioutput='raw_values')
        rmse_test_list = root_mean_squared_error(y_test, y_test_pred, multioutput='raw_values')

        rmse_train_average = root_mean_squared_error(y_train, y_train_pred)
        rmse_test_average = root_mean_squared_error(y_test, y_test_pred)

        std_targets = np.std(targets, axis=0)

        nrmse_train_list = rmse_train_list / std_targets
        nrmse_test_list = rmse_test_list / std_targets

        nrmse_train_average = np.mean(nrmse_train_list)
        nrmse_test_average = np.mean(nrmse_test_list)

        return rmse_train_average, rmse_test_average, rmse_train_list, rmse_test_list, nrmse_train_average, nrmse_test_average, nrmse_train_list, nrmse_test_list
