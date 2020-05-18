import tensorflow as tf

from data_processing import *
import train
from train import *
from MFE import plot_statistics

def plot_prediction_of_model(model_name, time_start=None):
    tf_model = tf.keras.models.load_model(model_name, compile=False)

    args = model_to_args(model_name)
    if args['phys'] == 'informed':
        tf_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=phys_loss)
    else:
        tf_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')

    time, data = load_data(args['case'], args['noise'])
    TRAIN_SPLIT = int(data.shape[0] * train.TRAIN_SPLIT / 100)
    VAL_SPLIT = int(data.shape[0] * train.VAL_SPLIT / 100)
    data = standardize_dataset(data)

    if args['case'] == 'MFE':
        # good peaks [95368, 259273]
        # large peaks at [ 68829  95368 261273]
        # meadium peaks at [ 31402  49627  68829  95368  97825 173286 203372 206545 261273]
        if time_start is None:
            time_start = 95368
        time_interval = 3000
    else:
        time_start = 1000
        time_interval = 1000

    # x_train, y_train = split_data(data, data, 0, TRAIN_SPLIT, past_history, future_target)
    x_val, y_val = split_data(data, data, time_start, time_start + time_interval, args['past_history'], time_interval)
    # x_test, y_test = split_data(data, data[:, 1], TRAIN_SPLIT + VAL_SPLIT, None, past_history, future_target, STEP)

    # train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    val_data = val_data.batch(BATCH_SIZE)
    for x, y in val_data.take(1):
        prediction = forecast(tf_model, x, args['past_history'], time_interval)
        plot_prediction(x[0], y[0], prediction, model_name)


def plot_statistics_of_model(model_name):
    tf_model = tf.keras.models.load_model(model_name, compile=False)

    args = model_to_args(model_name)
    if args['phys'] == 'informed':
        tf_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=phys_loss)
    else:
        tf_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')


    time, data = load_data(args['case'])
    TRAIN_SPLIT = int(data.shape[0] * train.TRAIN_SPLIT / 100)
    VAL_SPLIT = int(data.shape[0] * train.VAL_SPLIT / 100)
    data = standardize_dataset(data)

    if args['case'] == 'MFE':
        time_start = 75000
        time_interval = 5000
    else:
        time_start = 1000
        time_interval = 1000

    # x_train, y_train = split_data(data, data, 0, TRAIN_SPLIT, past_history, future_target)
    x_val, y_val = split_data(data, data, time_start, time_start+time_interval, args['past_history'], time_interval)
    # x_test, y_test = split_data(data, data[:, 1], TRAIN_SPLIT + VAL_SPLIT, None, past_history, future_target, STEP)

    # train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    val_data = val_data.batch(BATCH_SIZE)
    for x, y in val_data.take(1):
        prediction = forecast(tf_model, x, args['past_history'], time_interval)
        history = destandardize_dataset(x[0])
        true_future = destandardize_dataset(y[0])
        prediction = destandardize_dataset(prediction)
        plot_statistics(history, true_future, prediction, model_name)


def analyse_log_file(log_file):
    args = model_to_args(log_file)
    history = {}
    with open(log_file, 'r') as log_fp:
        for line in log_fp.readlines():
            elems = line.split(';')
            if len(elems) > 2 and elems[0].isdigit():
                try:
                    history[int(elems[0])] = {'loss': float(elems[1]),
                                              'val_loss': float(elems[2])}
                except ValueError:
                    continue


    to_plot_history = {'loss': [history[key]['loss'] for key in sorted(history.keys())],
                       'val_loss': [history[key]['val_loss'] for key in sorted(history.keys())]}

    plot_train_history(to_plot_history, log_file)
    return min(to_plot_history['val_loss'])/args['future_target']

def analyse_models(model_dir):
    results_log = {}
    for file in sorted(os.listdir(model_dir)):
        if file.split('.')[-1] == 'log':
            results_log[file] = analyse_log_file(os.path.join(model_dir, file))

        elif file.split('.')[-1] == 'h5':
            # plot_prediction_of_model(os.path.join(model_dir, file))
            pass

    print(results_log)
    print(min(results_log, key=results_log.get))

if __name__ == '__main__':
    pass

