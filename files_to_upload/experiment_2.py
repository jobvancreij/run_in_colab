from hyperopt import tpe, hp,fmin
from hyperopt.mongoexp import MongoTrials
import numpy as np
from hyperopt import Trials, STATUS_OK, tpe, STATUS_FAIL
from LJT_database.firestore_codes import  add_update,retrieve_updates
from LJT_database.merge_dataset import retrieve_data_predictors
from deep_learning_models.training_models import build_model
import sys
import warnings
import subprocess
from LJT_database.feature_prep import feature_preperation
import socket
from functools import partial
from LJT_helper_functions.helpers import send_message_telegram
import platform
from deep_learning_models.feature_selection import create_reduced_features

warnings.filterwarnings('ignore')
np.random.seed(38)
test = {}
hyperopt_values = "34.66.233.70:5000/eval_db/jobs"
def make_space(settings_experiment_2):
    """
    The defined settings are inserted in to the searchspace for hyperopt
    :param settings_experiment_2: settings
    :return: searchsspace filled in
    """
    space = {
        'optimizer': hp.choice("optimizer",
                               settings_experiment_2['optimizer']),
        'neurons': hp.choice("neurons",
                             settings_experiment_2['neurons']),
        'dropout': hp.uniform("dropout",
                              settings_experiment_2['dropout']['lowest_poss'],
                              settings_experiment_2['dropout']['highest_poss']),
        'loss_func': hp.choice("loss_func",
                          settings_experiment_2['loss_func']),
        'activation_function': hp.choice("activation_function",
                                         settings_experiment_2['activation_function']),
        'number_layers': hp.choice("number_layers",
                                   settings_experiment_2['number_layers']),
        'batch_size': hp.quniform("batch_size",
                                  settings_experiment_2['batch_size']['lowest_poss'],
                                  settings_experiment_2['batch_size']['highest_poss'],
                                  settings_experiment_2['batch_size']['increments']),
        'epochs': hp.quniform('epochs',
                              settings_experiment_2['epochs']['lowest_poss'],
                              settings_experiment_2['epochs']['highest_poss'],
                              settings_experiment_2['epochs']['increments']),
        'learning_rate': hp.loguniform('learning_rate',
                                   settings_experiment_2['learning_rate']['lowest_poss'],
                                    settings_experiment_2['learning_rate']['highest_poss']),
        "bidrectional": hp.choice("bidrectional",
                                  settings_experiment_2['bidirectional']),
        "window_size": hp.choice("window_size",
                                 settings_experiment_2['window']),
        "time_ahead_prediction": hp.choice("time_ahead_prediction",
                                           settings_experiment_2['prediction']),
    }
    return space



if __name__ == "__main__":
    if len(sys.argv) - 1 !=2:
        raise TypeError("Please give arguments. Arg 1 = coin, Arg2 = algorithm")
    coin = sys.argv[1].upper()
    algorithm = sys.argv[2].upper()
    settings = retrieve_updates(dataset=f"{coin}_{algorithm}_experiments",
                                        document="experiment_general_settings")
    df = feature_preperation(retrieve_data_predictors(settings)) #retrieve and clean
    df = create_reduced_features(df, settings) #get the features ater pca

    settings_experiment_2 = retrieve_updates(dataset=f"{coin}_{algorithm}_experiments",
                                   document="experiment2_settings")

    tpe_algo = tpe.suggest
    tpe_trials = MongoTrials(f"mongo://{hyperopt_values}",exp_key=settings["experiment_date"])

    space = make_space(settings_experiment_2)
    tpe_algo = tpe.suggest
    tpe_best = fmin(partial(build_model,settings=settings,dataset_training=df), #partial alllows for extra arguments
                    space=space,
                    algo=tpe_algo,
                    trials=tpe_trials,
                    max_evals=settings_experiment_2['number_experiments']
                   )
    #storage.terminate()
    print("done")
    print(tpe_best)
    send_message_telegram("Experiment 2",f"Experiment 2 is finished for {socket.gethostname()}")



