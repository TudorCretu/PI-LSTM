import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from scipy import interpolate

from data_processing import *
from MFE import model

tf.random.set_seed(8)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
model_dir = "models/"
TRAIN_SPLIT = 65
VAL_SPLIT = 20

# Get general arguments
if len(sys.argv) > 1:
    case = sys.argv[1]
    num_hidden = int(sys.argv[2])  # 32
    num_layers = int(sys.argv[3])  # 1
    epochs = int(sys.argv[4])  # 100
    step_mode = sys.argv[5]
    past_history =  int(sys.argv[6])
    if step_mode == 'multi':
        future_target = int(sys.argv[7])
    else:
        future_target = 1
    phys = sys.argv[8]
    noise = int(sys.argv[9])

else:
    case = 'Lorenz'
    num_hidden = 32  # 32
    num_layers = 2  # 1
    epochs = 10  # 100
    step_mode = 'multi'
    past_history = 250
    if step_mode == 'multi':
        future_target = 50
    else:
        future_target = 1
    phys = "informed"
    noise = 0

if case == 'Lorenz':
    data_min = np.array([-17.80792203, -23.81998559, 0.96173738])
    data_max = np.array([19.55509396, 27.18349319, 47.83402691])
    factor = 20
    dt = 0.01
else:
    data_min = np.array([0.0380113, -0.40291927, -0.35740569,
                         -0.27762828, -0.33655982, -0.33667415,
                         -0.34905171, -0.24684337, -0.49509102])
    data_max = np.array([0.85322096, 0.40385395, 0.23916417,
                         0.29868494, 0.26576593, 0.54522006,
                         0.37806215, 0.27541811, 0.01573427])
    factor = 0.05
    dt = 0.25

if phys == 'informed':
    if case == 'Lorenz':
        def system_step_tf(Uin, dt):
            sigma = 10.0
            rho = 28.0
            beta = 8/3
            x, y, z = tf.unstack(Uin, axis=2)
            x1 = x + dt * (sigma * (y - x))
            y1 = y + dt * (rho * x - y - tf.multiply(x, z))
            z1 = z + dt * ((tf.multiply(x, y) - beta * z))
            return tf.stack([x1, y1, z1], axis=2)
    else:
        def system_step_tf(Uin, dt):
            a1, a2, a3, a4, a5, a6, a7, a8, a9 = tf.unstack(Uin, axis=2)
            da = model([a1, a2, a3, a4, a5, a6, a7, a8, a9]) * dt
            a1 = a1 + da[0]
            a2 = a2 + da[1]
            a3 = a3 + da[2]
            a4 = a4 + da[3]
            a5 = a5 + da[4]
            a6 = a6 + da[5]
            a7 = a7 + da[6]
            a8 = a8 + da[7]
            a9 = a9 + da[8]
            return tf.stack([a1, a2, a3, a4, a5, a6, a7, a8, a9], axis=2)


def model_to_args(model_name):
    model_name = model_name.split('.')[0]
    model_name = model_name.split('/')[-1]
    args = model_name.split('_')
    case = args[1]
    num_hidden = int(args[2])  # 32
    num_layers = int(args[3])  # 1
    epochs = int(args[4])  # 100
    step_mode = args[5]
    past_history =  int(args[6])
    future_target = int(args[7])
    phys = args[8]
    if len(args)>9:
        noise = int(args[9])
    else:
        noise = 0
    return {
        'case': case,
        'num_hidden': num_hidden,
        'num_layers': num_layers,
        'epochs': epochs,
        'step_mode': step_mode,
        'past_history': past_history,
        'future_target': future_target,
        'phys': phys,
        'noise': noise
    }


def smooth_head(history, signal, width):
    history_index = list(range(-width, 0))
    signal_index = list(range(width, 2*width))
    interp_index = list(range(0, width))
    y = np.concatenate((history[:, history_index, :], signal[:, signal_index, :]), axis=1)
    x = np.array(history_index + signal_index)
    f = interpolate.interp1d(x, y, kind='quadratic', axis=1)
    interp = f(interp_index)
    signal[:, interp_index, :] = interp
    return signal


def forecast(model, history, n_input, duration, with_smoothing=False):
    forecast = []
    data = np.array(history)

    offset = 0
    if with_smoothing:
        width = 3

    while len(forecast) < duration:
        input_x = data[:, -n_input:, :]
        yhat = model.predict(input_x, verbose=0)[:, :, :]
        if with_smoothing:
            yhat = smooth_head(data, yhat, width)

        data = np.concatenate((data, yhat[:, :, :]), axis=1)
        forecast.extend(yhat[0][offset:])

    return forecast[:duration]


def create_time_steps(length):
    return np.array(list(range(-length, 0)))


def rescale(array, case):
    if case == 'Lorenz':
        array = array*0.01*0.934
    else:
        array = array*0.25*0.0096
    return array


def plot_prediction(history, true_future, prediction, model_name=None):
    num_in = create_time_steps(len(history))
    num_out = np.arange(len(true_future.numpy()))

    if model_name:
        case = model_to_args(model_name)['case']
    num_in = rescale(num_in, case)
    num_out = rescale(num_out, case)

    if history.shape[1] == 3:
        fig, axs = plt.subplots(nrows=history.shape[1], figsize=(10, 7))
        for i, ax in enumerate(axs):
            ax.plot(num_in, np.array(history[:, i]), label='History')
            ax.plot(num_out, np.array(true_future)[:, i], label='True Future')
            ax.plot(num_out, np.array(prediction)[:, i], label='Predicted Future')
            ax.set_ylabel("$a_" + str(i + 1) + "$")
            ax.legend(loc='upper left')
    else:
        fig, axs = plt.subplots(nrows=9, ncols=1, figsize=(10, 20))
        gs1 = gridspec.GridSpec(9, 1)

        for i in range(9):
            ax = plt.subplot(gs1[i])
            ax.plot(num_in, np.array(history[:, i]), label='History')
            ax.plot(num_out, np.array(true_future)[:, i], label='True Future')
            ax.plot(num_out, np.array(prediction)[:, i], label='Predicted Future')
            ax.set_ylabel("$a_" + str(i + 1) + "$")
            ax.legend(loc='upper left')
    plt.xlabel(r'$\lambda_{max}$t', fontsize=14)
    if model_name:
        fig.suptitle(model_name, fontsize=16)
    plt.show()


def plot_train_history(history, model_name=None):
    loss = history['loss']
    val_loss = history['val_loss']

    plt.figure()

    plt.plot(range(len(loss)), loss, label='Training loss')
    plt.plot(range(len(val_loss)), val_loss, label='Validation loss')
    if model_name:
        plt.title('Training and validation loss of ' + model_name)
    else:
        plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def phys_loss(y_true, y_pred):
    loss_data = tf.keras.losses.MSE(y_true, y_pred)

    y_pred = destandardize_dataset(y_pred)

    Yval_exact_tf = system_step_tf(y_pred[:, :-1, :], dt)

    loss_phys = tf.reduce_mean(tf.reduce_mean(tf.square(y_pred[:, 1:, :] - Yval_exact_tf), 1))

    return loss_data + loss_phys/factor


BATCH_SIZE = 256
BUFFER_SIZE = 10000


def standardize_dataset(data):
    data = (data - data_min) / (data_max - data_min)
    return data


def destandardize_dataset(data):
    return data * (data_max - data_min) + data_min


if __name__ == '__main__':
    time, data = load_data(case, noise)
    TRAIN_SPLIT = int(data.shape[0] * TRAIN_SPLIT / 100)
    VAL_SPLIT = int(data.shape[0] * VAL_SPLIT / 100)
    data = standardize_dataset(data)
    x_train, y_train = split_data(data, data, 0, TRAIN_SPLIT, past_history, future_target)
    x_val, y_val = split_data(data, data, TRAIN_SPLIT, TRAIN_SPLIT+VAL_SPLIT, past_history, future_target)
    # x_test, y_test = split_data(data, data[:, 1], TRAIN_SPLIT + VAL_SPLIT, None, past_history, future_target, STEP)

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]

    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_data = val_data.batch(BATCH_SIZE)

    model_name = model_dir + "model_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(case,
                                                                       num_hidden,
                                                                       num_layers,
                                                                       epochs,
                                                                       step_mode,
                                                                       past_history,
                                                                       future_target,
                                                                       phys,
                                                                       noise)

    if os.path.exists(model_name+'.h5'):
        tf_model = tf.keras.models.load_model(model_name+'.h5', compile=False)
        if phys == 'informed':
            tf_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=phys_loss)
        else:
            tf_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
    else:
        tf_model = tf.keras.models.Sequential()

        tf_model.add(tf.keras.layers.LSTM(num_hidden,
                                          input_shape=(n_timesteps, n_features)))

        tf_model.add(tf.keras.layers.RepeatVector(future_target))
        for _ in range(num_layers - 1):
            tf_model.add(tf.keras.layers.LSTM(num_hidden, return_sequences=True, activation='elu'))
        tf_model.add(tf.keras.layers.LSTM(num_hidden, return_sequences=True, activation='elu'))

        tf_model.add(tf.keras.layers.Dense(num_hidden, activation='elu'))
        tf_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(x_train.shape[2], activation='elu')))

        if phys == 'informed':
            tf_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=phys_loss)
        else:
            tf_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')

    es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=50)
    mc = tf.keras.callbacks.ModelCheckpoint(model_name+'.h5', monitor='val_loss', mode='min', save_best_only=True)
    tnan = tf.keras.callbacks.TerminateOnNaN()
    log = tf.keras.callbacks.CSVLogger(model_name+'.log', append=True, separator=';')
    fit_history = tf_model.fit(train_data,
                               epochs=epochs,
                               validation_data=val_data,
                               callbacks=[es, mc, tnan, log],
                               verbose=0)
