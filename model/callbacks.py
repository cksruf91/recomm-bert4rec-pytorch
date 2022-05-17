import os
import sqlite3

import mlflow
import numpy as np


class ModelCheckPoint:

    def __init__(self, file, save_best=True, monitor='val_loss', mode='min'):
        self.file = file
        save_dir = os.path.dirname(self.file)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode
        init_values = {'min': np.inf, 'max': -np.inf}
        self.best_score = init_values[mode]

    def __call__(self, model, history):
        val_score = history[self.monitor]
        check_point = self.file.format(**history)

        if not self.save_best:
            self.save_model(model, check_point)
        elif self._best(val_score, self.best_score):
            self.best_score = val_score
            self.save_model(model, check_point)

    def _best(self, val, best):
        if self.mode == 'min':
            return val <= best
        else:
            return val >= best

    @staticmethod
    def save_model(model, file_name):
        model.save(file_name)


class TrainHistory:

    def __init__(self, file):
        self.file = file
        if os.path.isfile(self.file):
            with open(self.file, 'a') as f:
                f.write('\n')

    def __call__(self, model, history):
        with open(self.file, 'a+') as f:
            f.write(str(history) + '\n')


class MlflowLogger:

    def __init__(self, experiment_name: str, model_params: dict, run_name=None, run_id=None, log_model=False,
                 model_name='torch_model', monitor='val_loss', mode='min'):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.model_params = model_params
        self.log_model = log_model
        self.model_name = model_name
        self._set_env()
        self.run_id = self._get_run_id() if run_id is None else run_id

        init_values = {'min': np.inf, 'max': -np.inf}
        self.mode = mode
        self.monitor = monitor
        self.best_score = init_values[self.mode]

    def __call__(self, model, history):
        with mlflow.start_run(run_id=self.run_id):
            if self.log_model:
                if self._best(history[self.monitor], self.best_score):
                    self._truncate_log_model_history()
                    mlflow.pytorch.log_model(
                        pytorch_model=model,
                        artifact_path=self.model_name
                    )
                    self.best_score = history[self.monitor]

            mlflow.log_metrics(history, step=history['epoch'])

    def __eq__(self, other):
        return "MLFlow" == other

    def _get_run_id(self):
        with mlflow.start_run(run_name=self.run_name) as mlflow_run:
            mlflow.log_params(self.model_params)
            run_id = mlflow_run.info.run_id
        return run_id

    def _set_env(self):
        if os.getenv('MLFLOW_TRACKING_URI') is None:
            raise ValueError("Environment variable MLFLOW_TRACKING_URI is not exist")

        mlflow.set_experiment(self.experiment_name)

    def _best(self, val, best):
        if self.mode == 'min':
            return val <= best
        else:
            return val >= best

    def _truncate_log_model_history(self):
        """
        On each epoch mlflow.pytorch.log_model append log to 'mlflow.log-model.history' tag
        after several epoch this tag becomes very large and exceeds the length limit.
        so manually truncate log from tags table issues
        reference github issues : https://github.com/mlflow/mlflow/issues/3032
        """
        sql = f'''
        UPDATE tags SET value = '[]' 
         WHERE run_uuid = '{self.run_id}'
           AND key = 'mlflow.log-model.history'
        '''
        with sqlite3.connect('/home/ec2-user/mydb.sqlite') as conn:  # sqlite:///mydb.sqlite
            cur = conn.cursor()
            cur.execute(sql)
