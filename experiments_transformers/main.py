import os
import argparse
import json
import itertools
import time
import requests
from multiprocessing import Process, Manager
import numpy as np
import pandas as pd
from metrics import METRICS, evaluate
from preprocessing import denormalize




def check_params(datasets, models, results_path, parameters, metrics, csv_filename):
    assert len(datasets) > 0, "dataset parameter is not well defined."
    assert all(
        os.path.exists(ds_path) for ds_path in datasets
    ), "dataset paths are not well defined."
    assert all(
        param in parameters.keys()
        for param in [
            "normalization_method",
            "past_history_factor",
            "batch_size",
            "epochs",
            "max_steps_per_epoch",
            "learning_rate",
            "model_params",
        ]
    ), "Some parameters are missing in the parameters file."
    assert all(
        model in parameters["model_params"] for model in models
    ), "models parameter is not well defined."
    assert metrics is None or all(m in METRICS.keys() for m in metrics)


def read_results_file(csv_filepath, metrics):

    try:
        results = pd.read_csv(csv_filepath, sep=";", index_col=0)
    except IOError:
        results = pd.DataFrame(
            columns=[
                "DATASET",
                "MODEL",
                "MODEL_INDEX",
                "MODEL_DESCRIPTION",
                "FORECAST_HORIZON",
                "PAST_HISTORY_FACTOR",
                "PAST_HISTORY",
                "BATCH_SIZE",
                "EPOCHS",
                "STEPS",
                "OPTIMIZER",
                "LEARNING_RATE",
                "NORMALIZATION",
                "TEST_TIME",
                "TRAINING_TIME",
                *metrics ,
                "LOSS",
                "VAL_LOSS",
            ]
        )
    return results


def product(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def read_data(dataset_path, normalization_method, past_history_factor):
    # read normalization params
    norm_params = None
    with open(
        os.path.normpath(dataset_path)
        + "/{}/norm_params.json".format(normalization_method),
        "r",
    ) as read_file:
        norm_params = json.load(read_file)

    # read training / validation data
    tmp_data_path = os.path.normpath(dataset_path) + "/{}/{}/".format(
        normalization_method, past_history_factor
    )

    x_train = np.load(tmp_data_path + "x_train.np.npy")
    y_train = np.load(tmp_data_path + "y_train.np.npy")
    x_test = np.load(tmp_data_path + "x_test.np.npy")
    y_test = np.load(tmp_data_path + "y_test.np.npy")
    if os.path.isfile(tmp_data_path + "invalidParams.np.npy"):
        invalidParams = np.load(tmp_data_path + "invalidParams.np.npy")
        norm_params = np.delete(norm_params,invalidParams)
        

    y_test_denorm = np.asarray(
        [
            denormalize(y_test[i], norm_params[i], normalization_method)
            for i in range(y_test.shape[0])
        ]
    )

    print("TRAINING DATA")
    print("Input shape", x_train.shape)
    print("Output_shape", y_train.shape)
    print("TEST DATA")
    print("Input shape", x_test.shape)
    print("Output_shape", y_test.shape)

    return x_train, y_train, x_test, y_test, y_test_denorm, norm_params



def _run_experiment_transformer(
    gpu_device,
    dataset,
    dataset_path,
    results_path,
    csv_filepath,
    metrics,
    epochs,
    normalization_method,
    past_history_factor,
    max_steps_per_epoch,
    batch_size,
    learning_rate,
    model_name,
    model_index,
    model_args,
):
    print("Start _run_experiment_transformer")
    import gc
    from models import create_model

    import torch
    from pytorch_lightning import Trainer, seed_everything
    from torch.utils.data import DataLoader, TensorDataset
 

    results = read_results_file(csv_filepath, metrics)

    x_train, y_train, x_test, y_test, y_test_denorm, norm_params = read_data(
        dataset_path, normalization_method, past_history_factor
    )
    
    forecast_horizon = y_test.shape[1]
    past_history = x_test.shape[1]


    steps_per_epoch = min(
        int(np.ceil(x_train.shape[0] / batch_size)), max_steps_per_epoch,
    )
    

    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float().unsqueeze(-1)

    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float().unsqueeze(-1)
    


    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, 
                                  batch_size = batch_size, 
                                  shuffle = True)

    val_loader = DataLoader(val_dataset, 
                                batch_size = batch_size, 
                                shuffle = False)

    test_loader = DataLoader(val_dataset, 
                                batch_size = 256, 
                                shuffle = False)


    #seed_everything(42,workers=true)


    trainer = Trainer(
        max_epochs=epochs,
        max_steps=steps_per_epoch, 
        gpus=[gpu_device],
        checkpoint_callback=False
    )

    print("Creating model")
   
    model = create_model(
        model_name,
        x_train.shape,
        output_size=forecast_horizon,
        **model_args
    )



    training_time_0 = time.time()
    trainer.fit(model,train_loader,val_loader)
    training_time = time.time() - training_time_0

    print("End training")

    train_loss = float(trainer.callback_metrics["train_loss"].to("cpu"))
    val_loss = float(trainer.callback_metrics["val_loss"].to("cpu"))
    print("loss saved")


    def predictMultiStepRegresive(model,x_test, nSteps):
        '''
        Inference for autoregresive models
        '''
        with torch.no_grad():
            encoderInput = x_test
            decoderInput = x_test[:, -1,:]
            decoderInput = decoderInput.unsqueeze(-1)
            for _ in range(nSteps): # We append the prediction on step 0 to the original input to get the input for step 1 and so on.
                out = model.forward(encoderInput,decoderInput)  # We don't need mask for evaluation, we are not giving the model any future input.
                lastPrediction = out[:,-1].detach() #detach() is quite important, otherwise we will keep the variable "out" in memory and cause an out of memory error.
                lastPrediction = lastPrediction.unsqueeze(-1)
                decoderInput = torch.cat((decoderInput,lastPrediction),1) 

            decoderInput = decoderInput.squeeze(-1)
        return decoderInput[:,1:]

    def predictMultiStepRegresive2(model,x_test, nSteps):
        '''
        Inference for autoregresive models
        '''
        with torch.no_grad():
            z = x_test
            for _ in range(nSteps): # We append the prediction on step 0 to the original input to get the input for step 1 and so on.
                out = model.forward(z)  # We don't need mask for evaluation, we are not giving the model any future input.
                lastPrediction = out[:,-1].detach() #detach() is quite important, otherwise we will keep the variable "out" in memory and cause an out of memory error.
                lastPrediction = lastPrediction.unsqueeze(-1)
                z = torch.cat((z,lastPrediction),1) 

            output = z[:,x_test.shape[1]:].squeeze(-1)
        return output

    
    # Only use "device" for inference, as Pytorch Lighting already handles it for training
    device = torch.device("cuda:" + str(gpu_device) if torch.cuda.is_available() else "cpu")


    model.to(device)
    model.eval()
    x_test = x_test.to(device)
    if model_name.endswith("AR"):
        '''
        Autoregresive inference
        '''
        test_time_0 = time.time()
        test_forecast = predictMultiStepRegresive2(model, x_test,y_test.shape[1])
        test_time = time.time() - test_time_0
    else:
        '''
        Non-Autoregresive inference
        '''

        test_time_0 = time.time()
        test_forecast = model(x_test)
        test_time = time.time() - test_time_0
    print("End test")
    test_forecast = test_forecast.detach().to("cpu").numpy()


    for i, nparams in enumerate(norm_params):
        test_forecast[i] = denormalize(
            test_forecast[i], nparams, method=normalization_method,
        )

    if metrics:
        test_metrics = evaluate(y_test_denorm, test_forecast, metrics)
        print(test_metrics)
    else:
        print("Metrics empty")
        test_metrics = {}

    # Save results
    predictions_path = "{}/{}/{}/{}/{}/{}/{}/{}/".format(
        results_path,
        dataset,
        normalization_method,
        past_history_factor,
        epochs,
        batch_size,
        learning_rate,
        model_name,
    )
    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path)
    np.save(
        predictions_path + str(model_index) + ".npy", test_forecast,
    )

    results = results.append(
        {
            "DATASET": dataset,
            "MODEL": model_name,
            "MODEL_INDEX": model_index,
            "MODEL_DESCRIPTION": str(model_args),
            "FORECAST_HORIZON": forecast_horizon,
            "PAST_HISTORY_FACTOR": past_history_factor,
            "PAST_HISTORY": past_history,
            "BATCH_SIZE": batch_size,
            "EPOCHS": epochs,
            "STEPS": steps_per_epoch,
            "OPTIMIZER": "CustomAdam",
            "LEARNING_RATE": learning_rate,
            "NORMALIZATION": normalization_method,
            "TEST_TIME": test_time,
            "TRAINING_TIME": training_time,
            **test_metrics,
            "LOSS": str(train_loss),
            "VAL_LOSS": str(val_loss),
        },
        ignore_index=True,
    )

    results.to_csv(
        csv_filepath, sep=";",
    )

    gc.collect()
    del model, x_train, x_test, y_train, y_test, y_test_denorm, test_forecast




def run_experiment_transformer(
    error_dict,
    gpu_device,
    dataset,
    dataset_path,
    results_path,
    csv_filepath,
    metrics,
    epochs,
    normalization_method,
    past_history_factor,
    max_steps_per_epoch,
    batch_size,
    learning_rate,
    model_name,
    model_index,
    model_args,
):
    print("Start run_experiment_transformer")
    try:
        
        _run_experiment_transformer(
            gpu_device,
            dataset,
            dataset_path,
            results_path,
            csv_filepath,
            metrics,
            epochs,
            normalization_method,
            past_history_factor,
            max_steps_per_epoch,
            batch_size,
            learning_rate,
            model_name,
            model_index,
            model_args,
        )
    except Exception as e:
        error_dict["status"] = -1
        error_dict["message"] = str(e)
    else:
        error_dict["status"] = 1

def _run_experiment(
    gpu_device,
    dataset,
    dataset_path,
    results_path,
    csv_filepath,
    metrics,
    epochs,
    normalization_method,
    past_history_factor,
    max_steps_per_epoch,
    batch_size,
    learning_rate,
    model_name,
    model_index,
    model_args,
):
    import gc
    import tensorflow as tf
    from models import create_model

    tf.keras.backend.clear_session()

    def select_gpu_device(gpu_number):
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if len(gpus) >= 2 and gpu_number is not None:
            device = gpus[gpu_number]
            tf.config.experimental.set_memory_growth(device, True)
            tf.config.experimental.set_visible_devices(device, "GPU")

    select_gpu_device(gpu_device)

    results = read_results_file(csv_filepath, metrics)

    x_train, y_train, x_test, y_test, y_test_denorm, norm_params = read_data(
        dataset_path, normalization_method, past_history_factor
    )
    x_train = tf.convert_to_tensor(x_train)
    y_train = tf.convert_to_tensor(y_train)
    x_test = tf.convert_to_tensor(x_test)
    y_test = tf.convert_to_tensor(y_test)
    y_test_denorm = tf.convert_to_tensor(y_test_denorm)

    forecast_horizon = y_test.shape[1]
    past_history = x_test.shape[1]
    steps_per_epoch = min(
        int(np.ceil(x_train.shape[0] / batch_size)), max_steps_per_epoch,
    )

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    model = create_model(
        model_name,
        x_train.shape,
        output_size=forecast_horizon,
        optimizer=optimizer,
        loss="mae",
        **model_args
    )
    print(model.summary())

    training_time_0 = time.time()
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=(x_test, y_test),
        shuffle=True,
    )
    training_time = time.time() - training_time_0

    # Get validation metrics
    test_time_0 = time.time()
    test_forecast = model(x_test).numpy()
    test_time = time.time() - test_time_0

    normalized_test_forecast = test_forecast.copy()

    for i, nparams in enumerate(norm_params):
        test_forecast[i] = denormalize(
            test_forecast[i], nparams, method=normalization_method,
        )
    if metrics:
        test_metrics = evaluate(y_test_denorm, test_forecast, metrics)
    else:
        test_metrics = {}

    # Save results
    predictions_path = "{}/{}/{}/{}/{}/{}/{}/{}/".format(
        results_path,
        dataset,
        normalization_method,
        past_history_factor,
        epochs,
        batch_size,
        learning_rate,
        model_name,
    )
    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path)
    np.save(
        predictions_path + str(model_index) + ".npy", test_forecast,
    )
    np.save(
        predictions_path + "Normalize" +  str(model_index) + ".npy", normalized_test_forecast,
    )
    results = results.append(
        {
            "DATASET": dataset,
            "MODEL": model_name,
            "MODEL_INDEX": model_index,
            "MODEL_DESCRIPTION": str(model_args),
            "FORECAST_HORIZON": forecast_horizon,
            "PAST_HISTORY_FACTOR": past_history_factor,
            "PAST_HISTORY": past_history,
            "BATCH_SIZE": batch_size,
            "EPOCHS": epochs,
            "STEPS": steps_per_epoch,
            "OPTIMIZER": "Adam",
            "LEARNING_RATE": learning_rate,
            "NORMALIZATION": normalization_method,
            "TEST_TIME": test_time,
            "TRAINING_TIME": training_time,
            **test_metrics,
            "LOSS": str(history.history["loss"][-1]),
            "VAL_LOSS": str(history.history["val_loss"][-1]),
        },
        ignore_index=True,
    )

    results.to_csv(
        csv_filepath, sep=";",
    )

    gc.collect()
    del model, x_train, x_test, y_train, y_test, y_test_denorm, test_forecast


def run_experiment(
    error_dict,
    gpu_device,
    dataset,
    dataset_path,
    results_path,
    csv_filepath,
    metrics,
    epochs,
    normalization_method,
    past_history_factor,
    max_steps_per_epoch,
    batch_size,
    learning_rate,
    model_name,
    model_index,
    model_args,
):
    try:
        _run_experiment(
            gpu_device,
            dataset,
            dataset_path,
            results_path,
            csv_filepath,
            metrics,
            epochs,
            normalization_method,
            past_history_factor,
            max_steps_per_epoch,
            batch_size,
            learning_rate,
            model_name,
            model_index,
            model_args,
        )
    except Exception as e:
        error_dict["status"] = -1
        error_dict["message"] = str(e)
    else:
        error_dict["status"] = 1

def main(args):
    datasets = args.datasets
    models = args.models
    results_path = args.output
    gpu_device = args.gpu
    metrics = args.metrics
    csv_filename = args.csv_filename

    parameters = None
    with open(args.parameters, "r") as params_file:
        parameters = json.load(params_file)

    check_params(datasets, models, results_path, parameters, metrics, csv_filename)

    if len(models) == 0:
        models = list(parameters["model_params"].keys())

    if metrics is None:
        metrics = list(METRICS.keys())

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    for dataset_index, dataset_path in enumerate(datasets):
        dataset = os.path.basename(os.path.normpath(dataset_path))

        csv_filepath = results_path + "/{}/{}".format(dataset, csv_filename)
        results = read_results_file(csv_filepath, metrics)
        current_index = results.shape[0]
        print("CURRENT INDEX", current_index)

        experiments_index = 0
        num_total_experiments = np.prod(
            [len(parameters[k]) for k in parameters.keys() if k != "model_params"]
            + [
                np.sum(
                    [
                        np.prod(
                            [
                                len(parameters["model_params"][m][k])
                                for k in parameters["model_params"][m].keys()
                            ]
                        )
                        for m in models
                    ]
                )
            ]
        )

        for epochs, normalization_method, past_history_factor in itertools.product(
            parameters["epochs"],
            parameters["normalization_method"],
            parameters["past_history_factor"],
        ):
            for batch_size, learning_rate in itertools.product(
                parameters["batch_size"], parameters["learning_rate"],
            ):
                for model_name in models:
                    for model_index, model_args in enumerate(
                        product(**parameters["model_params"][model_name])
                    ):
                        
                        experiments_index += 1
                        
                        if experiments_index <= current_index:
                            continue
                        # Run each experiment in a new Process to avoid GPU memory leaks
                        manager = Manager()
                        
                        error_dict = manager.dict()


                        # Run in Pytorch Lightning? Else Run in Tensorflow
                        if model_name.startswith("tr"):
                        
                            p = Process(
                                target=run_experiment_transformer,
                                args=(
                                    error_dict,
                                    gpu_device,
                                    dataset,
                                    dataset_path,
                                    results_path,
                                    csv_filepath,
                                    metrics,
                                    epochs,
                                    normalization_method,
                                    past_history_factor,
                                    parameters["max_steps_per_epoch"][0],
                                    batch_size,
                                    "Noam", #Custom learning rate
                                    model_name,
                                    model_index,
                                    model_args,
                                ),
                            )
                            p.start()
                            p.join()
                        else:

                            p = Process(
                                target=run_experiment,
                                args=(
                                    error_dict,
                                    gpu_device,
                                    dataset,
                                    dataset_path,
                                    results_path,
                                    csv_filepath,
                                    metrics,
                                    epochs,
                                    normalization_method,
                                    past_history_factor,
                                    parameters["max_steps_per_epoch"][0],
                                    batch_size,
                                    learning_rate,
                                    model_name,
                                    model_index,
                                    model_args,
                                ),
                            )
                            p.start()
                            p.join()


                        
                      

                        assert error_dict["status"] == 1, error_dict["message"]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        default=[],
        help="Dataset path to experiment over (separated by comma)",
    )
    parser.add_argument(
        "-m",
        "--models",
        nargs="*",
        default=[],
        help="Models to experiment over (separated by comma)",
    )
    parser.add_argument(
        "-p", "--parameters", help="Parameters file path",
    )
    parser.add_argument(
        "-o", "--output", default="./results", help="Output path",
    )
    parser.add_argument(
        "-c", "--csv_filename", default="results.csv", help="Output csv filename",
    )
    parser.add_argument("-g", "--gpu", type=int, default=None, help="GPU device")
    parser.add_argument(
        "-s",
        "--metrics",
        nargs="*",
        default=None,
        help="Metrics to use for evaluation. If not define it will use all possible metrics.",
    )
    args = parser.parse_args()
    
    main(args)
