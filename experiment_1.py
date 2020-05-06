from deep_learning_models.training_models import build_model
import sys
from LJT_database.firestore_codes import  add_update,retrieve_updates
import platform
from LJT_helper_functions.helpers import send_message_telegram, convert_datatype
from LJT_database.merge_dataset import retrieve_data_predictors
from LJT_database.update_table import insert_rows
from LJT_database.feature_prep import feature_preperation
import timeit
import socket
import time
def extract_new_experiment(coin,algorithm):
    try:
        experiments = retrieve_updates(dataset=f"{coin}_{algorithm}_experiments",
                                        document="experiment1_settings"
                                        )
        if 'windows' in platform.platform().lower(): #let windows take the easier ones. Make sure server does not
            windows = True
        else:
            windows = False
        all_experiments = experiments.keys()
        print(f"{len(all_experiments)} experiments left")
        if windows: #choose easiest experiment for windows
            experiment_chosen = min(int(experiment.split("_")[1]) for experiment in all_experiments)
        else: #choose most difficult experiment for other
            experiment_chosen = max(int(experiment.split("_")[1]) for experiment in all_experiments)
        print(f"Experiment chosen: {experiment_chosen}")
        #window = experiments[f"experiment_{experiment_chosen}"]['window']
       # prediction_ahead = experiments[f"experiment_{experiment_chosen}"]['prediction_ahead']
        current_experiment = experiments[f"experiment_{experiment_chosen}"]
        del experiments[f"experiment_{experiment_chosen}"]
        add_update(dataset=f"{coin}_{algorithm}_experiments", updates=experiments, document="experiment1_settings")
        return current_experiment
    except Exception as e:
        print(e)
        return False

def add_cancelled_experiment(coin,algorithm,current_experiment):
    """
    If a laptop/computer is closed this code updates the firestore experiments with the experiment that is not finshed.
    :param coin: which coin
    :param algorithm: LSTM / or GRU
    :param current_experiment: what is the interrupted experiment
    :return:
    """
    try:
        start = timeit.timeit()
        experiments = retrieve_updates(dataset=f"{coin}_{algorithm}_experiments",
                                        document="experiment1_settings"
                                        )
        if 'windows' in platform.platform().lower(): #let windows take the easier ones. Make sure server does not
            windows = True
        else:
            windows = False
        all_experiments = experiments.keys()
        if windows: #choose easiest experiment for windows
            experiment_number = min(int(experiment.split("_")[1]) for experiment in all_experiments) - 1
        else: #choose most difficult experiment for other
            experiment_number = max(int(experiment.split("_")[1]) for experiment in all_experiments) +1
        print(f"Update experiments with current as number: {experiment_number}")
        experiments[f"experiment_{experiment_number}"] = current_experiment  # choose random high number so unique
        add_update(dataset=f"{coin}_{algorithm}_experiments", updates=experiments, document="experiment1_settings")
        print("Succesfully added experiments to file in firestore, time in between = ",timeit.timeit() - start)
        return current_experiment
    except Exception as e:
        print(e)
        return False




if __name__ == "__main__":
    if len(sys.argv) - 1 !=2:
        raise TypeError("Please give arguments. Arg 1 = coin, Arg2 = algorithm")
    coin = sys.argv[1].upper()
    algorithm = sys.argv[2].upper()
    settings = retrieve_updates(dataset=f"{coin}_{algorithm}_experiments",
                                        document="experiment_general_settings")
    df = feature_preperation(retrieve_data_predictors(settings)) #retrieve and clean

    while True:
        try:
            experiment_chosen = extract_new_experiment(coin,algorithm)
            if experiment_chosen == False:
                break
            scores = build_model(experiment_chosen,settings,df)
            data_to_store = {**scores,**experiment_chosen}
            data_to_store['experiment'] = "experiment_1"
            data_to_store["experiment_date"] = settings["experiment_date"]
            for key,value in data_to_store.items(): #make sure that they have the correct datatype before inserting
                data_to_store[key] = convert_datatype(key,value)
            insert_rows(coinname=settings["prediction_dataset_name"],type_update=settings['prediction_database_name'],dataset=[data_to_store])
            print(f"Run finished, inserted: {data_to_store}")
        except KeyboardInterrupt:
            print("Training models interrupted")
            print('start adding current experiment to the list of experiments')
            add_cancelled_experiment(coin,algorithm,experiment_chosen)
            send_message_telegram("Experiment 1", f"Experiment 1 is stopped for {socket.gethostname()}")
            raise TypeError("Quitting now ")




    send_message_telegram("Experiment 1",f"Experiment 1 is finished for {socket.gethostname()}")
