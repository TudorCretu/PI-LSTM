import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py

Lx = 1.75 * np.pi
Lz = 1.2 * np.pi
Re = 600

X = 0
Y = 1
Z = 2

alpha = 2 * np.pi / Lx
beta = np.pi / 2
gamma = 2 * np.pi / Lz

Kay = np.sqrt(alpha ** 2 + gamma ** 2)
Kby = np.sqrt(beta ** 2 + gamma ** 2)
Kaby = np.sqrt(alpha ** 2 + beta ** 2 + gamma ** 2)

N8 = 2 * np.sqrt(2) / np.sqrt((alpha ** 2 + gamma ** 2) * (4 * alpha ** 2 + 4 * gamma ** 2 + np.pi ** 2))

# Domain is 0 < x < Lx ; -1 < y < 1; 0 < z < Lz


def da1(a):
    return beta ** 2 / Re - beta ** 2 / Re * a[0] - np.sqrt(3 / 2) * beta * gamma / Kaby * a[5] * a[7] + np.sqrt(
        3 / 2) * beta * gamma / Kby * a[1] * a[2]


def da2(a):
    return -(4 * beta ** 2 / 3 + gamma ** 2) * a[1] / Re + 5 * np.sqrt(2) * gamma ** 2 / (3 * np.sqrt(3) * Kay) * a[3] \
           * a[5] - \
           gamma ** 2 / (np.sqrt(6) * Kay) * a[4] * a[6] - alpha * beta * gamma / (np.sqrt(6) * Kay * Kaby) * a[4] * a[
               7] - \
           np.sqrt(3 / 2) * beta * gamma / Kby * (a[0] * a[2] + a[2] * a[8])


def da3(a):
    return -(beta ** 2 + gamma ** 2) / Re * a[2] + 2 / np.sqrt(6) * alpha * beta * gamma / (Kay * Kby) * (
                a[3] * a[6] + a[4] * a[5]) + \
           (beta ** 2 * (3 * alpha ** 2 + gamma ** 2) - 3 * gamma ** 2 * (alpha ** 2 + gamma ** 2)) / (
                       np.sqrt(6) * Kaby * Kby * Kay) * a[3] * a[7]


def da4(a):
    return -(3 * alpha ** 2 + 4 * beta ** 2) / (3 * Re) * a[3] - alpha / np.sqrt(6) * a[0] * a[4] - 10 / (
                3 * np.sqrt(6)) * alpha ** 2 / Kay * a[1] * a[5] - \
           np.sqrt(3 / 2) * alpha * beta * gamma / (Kay * Kby) * a[2] * a[6] - np.sqrt(
        3 / 2) * alpha ** 2 * beta ** 2 / (Kay * Kby * Kaby) * a[2] * a[7] - \
           alpha / np.sqrt(6) * a[4] * a[8]


def da5(a):
    return -(alpha ** 2 + beta ** 2) / Re * a[4] + alpha / np.sqrt(6) * a[0] * a[3] + alpha ** 2 / (np.sqrt(6) * Kay) \
           * a[1] * a[6] - \
           alpha * beta * gamma / (np.sqrt(6) * Kay * Kaby) * a[1] * a[7] + alpha / np.sqrt(6) * a[3] * a[
               8] + 2 / np.sqrt(6) * alpha * beta * gamma / (Kay * Kby) * a[2] * a[5]


def da6(a):
    return -(3 * alpha ** 2 + 4 * beta ** 2 + 3 * gamma ** 2) / (3 * Re) * a[5] + alpha / np.sqrt(6) * a[0] * a[6] + \
           np.sqrt(3 / 2) * beta * gamma / Kaby * a[0] * a[7] + 10 / (3 * np.sqrt(6)) * (
                       alpha ** 2 - gamma ** 2) / Kay * a[1] * a[3] - \
           2 * np.sqrt(2 / 3) * alpha * beta * gamma / (Kay * Kby) * a[2] * a[4] + alpha / np.sqrt(6) * a[6] * a[
               8] + np.sqrt(3 / 2) * beta * gamma / Kaby * a[7] * a[8]


def da7(a):
    return -(alpha ** 2 + beta ** 2 + gamma ** 2) / Re * a[6] - alpha / np.sqrt(6) * (a[0] * a[5] + a[5] * a[8]) + \
           np.sqrt(1 / 6) * (gamma ** 2 - alpha ** 2) / Kay * a[1] * a[4] + np.sqrt(1 / 6) * alpha * beta * gamma / (
                       Kay * Kby) * a[2] * a[3]


def da8(a):
    return -(alpha ** 2 + beta ** 2 + gamma ** 2) / Re * a[7] + 2 / np.sqrt(6) * alpha * beta * gamma / (Kay * Kaby) * \
           a[1] * a[4] + \
           gamma ** 2 * (3 * alpha ** 2 - beta ** 2 + 3 * gamma ** 2) / (np.sqrt(6) * Kay * Kby * Kaby) * a[2] * a[3]


def da9(a):
    return -9 * beta ** 2 / Re * a[8] + np.sqrt(3 / 2) * beta * gamma / Kby * a[1] * a[2] - np.sqrt(
        3 / 2) * beta * gamma / Kaby * a[5] * a[7]


def model(a):
    return np.array([da1(a),
                     da2(a),
                     da3(a),
                     da4(a),
                     da5(a),
                     da6(a),
                     da7(a),
                     da8(a),
                     da9(a)])


def u1(p):
    return np.array([np.sqrt(2) * np.sin(np.pi * p[Y] / 2),
                     0,
                     0])


def u2(p):
    return np.array([4/np.sqrt(3) * np.cos(np.pi * p[Y] / 2)**2 * np.cos(gamma*p[Z]),
                     0,
                     0])


def u3(p):
    return 2/np.sqrt(4 * gamma**2 + np.pi**2) * np.array([0,
                                                         2 * gamma * np.cos(np.pi * p[Y] / 2) * np.cos(gamma*p[Z]),
                                                         np.pi * np.sin(np.pi * p[Y] / 2) * np.sin(gamma * p[Z])])


def u4(p):
    return np.array([0,
                     0,
                     4/np.sqrt(3) * np.cos(alpha * p[X]) * np.cos(np.pi * p[Y] / 2)**2])


def u5(p):
    return np.array([0,
                     0,
                     2 * np.sin(alpha * p[X]) * np.sin(np.pi * p[Y] / 2)])


def u6(p):
    return 4*np.sqrt(2)/np.sqrt(3 * (alpha**2 + gamma**2)) * np.array([
                                                -gamma*np.cos(alpha*p[X])*np.cos(np.pi*p[Y]/2)**2*np.sin(gamma*p[Z]),
                                                0,
                                                alpha*np.sin(alpha*p[X])*np.cos(np.pi*p[Y]/2)**2*np.cos(gamma*p[Z])])


def u7(p):
    return 2*np.sqrt(2)/np.sqrt(alpha**2 + gamma**2) * np.array([
                                                gamma*np.sin(alpha*p[X])*np.sin(np.pi*p[Y]/2)*np.sin(gamma*p[Z]),
                                                0,
                                                alpha*np.cos(alpha*p[X])*np.sin(np.pi*p[Y]/2)*np.cos(gamma*p[Z])])


def u8(p):
    return N8 * np.array([np.pi * alpha * np.sin(alpha*p[X])*np.sin(np.pi*p[Y]/2)*np.sin(gamma*p[Z]),
                          2*(alpha**2 + gamma**2) * np.cos(alpha*p[X]) * np.cos(np.pi*p[Y]/2) * np.sin(gamma*p[Z]),
                          -np.pi * gamma * np.cos(alpha*p[X]) * np.sin(np.pi*p[Y]/2) * np.cos(gamma*p[Z])])


def u9(p):
    return np.array([np.sqrt(2) * np.sin(3 * np.pi * p[Y] / 2),
                     0,
                     0])


def make_grid(nx, ny, nz):
    x = np.linspace(0, Lx, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(0, Lz, nz)
    return x, y, z


def generate_u(x, y, z):
    u_0 = np.zeros([9, len(x), len(y), len(z), 3])
    for ix, px in enumerate(x):
        for iy, py in enumerate(y):
            for iz, pz in enumerate(z):
                u_0[0][ix][iy][iz] = u1([px, py, pz])
                u_0[1][ix][iy][iz] = u2([px, py, pz])
                u_0[2][ix][iy][iz] = u3([px, py, pz])
                u_0[3][ix][iy][iz] = u4([px, py, pz])
                u_0[4][ix][iy][iz] = u5([px, py, pz])
                u_0[5][ix][iy][iz] = u6([px, py, pz])
                u_0[6][ix][iy][iz] = u7([px, py, pz])
                u_0[7][ix][iy][iz] = u8([px, py, pz])
                u_0[8][ix][iy][iz] = u9([px, py, pz])
    return u_0


def calculate_velocities(x, y, z, a0, u_0):
    u = np.zeros([len(x), len(y), len(z), 3])
    for ix, px in enumerate(x):
        for iy, py in enumerate(y):
            for iz, pz in enumerate(z):
                u[ix][iy][iz] += a0[0] * u_0[0, ix, iy, iz]
                u[ix][iy][iz] += a0[1] * u_0[1, ix, iy, iz]
                u[ix][iy][iz] += a0[2] * u_0[2, ix, iy, iz]
                u[ix][iy][iz] += a0[3] * u_0[3, ix, iy, iz]
                u[ix][iy][iz] += a0[4] * u_0[4, ix, iy, iz]
                u[ix][iy][iz] += a0[5] * u_0[5, ix, iy, iz]
                u[ix][iy][iz] += a0[6] * u_0[6, ix, iy, iz]
                u[ix][iy][iz] += a0[7] * u_0[7, ix, iy, iz]
                u[ix][iy][iz] += a0[8] * u_0[8, ix, iy, iz]
    return u

def calculate_vorticity(x, y, z, u):
    w = np.zeros([len(u), len(x), len(y), len(z), 3])
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    dux_dy, dux_dz = np.gradient(u[:, :, :, :, X], dy, dz, axis=(2, 3))
    duy_dx, duy_dz = np.gradient(u[:, :, :, :, Y], dx, dz, axis=(1, 3))
    duz_dx, duz_dy = np.gradient(u[:, :, :, :, Z], dx, dy, axis=(1, 2))
    w[:, :, :, :, X] = duz_dy - duy_dz
    w[:, :, :, :, Y] = dux_dz - duz_dx
    w[:, :, :, :, Z] = duy_dx - dux_dy
    return w


def plot_mean_profile(a):
    x, y, z = make_grid(10, 100, 10)
    u_0 = generate_u(x, y, z)
    u = calculate_velocities(x, y, z, a, u_0)

    ux_mean = np.zeros([len(y)])
    for ix, px in enumerate(x):
        for iy, py in enumerate(y):
            for iz, pz in enumerate(z):
                ux_mean[iy] += u[ix][iy][iz][X]
    N = len(x) * len(z)

    ux_mean /= N

    axes = plt.gca()
    axes.set_xlim([-1, 1])
    axes.set_ylim([-1, 1])
    axes.set(xlabel="$u_x$", ylabel='y')
    axes.plot(ux_mean, y)
    plt.show()


def plot_statistics(history, true_future, prediction, model_name=None):
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
    gs1 = gridspec.GridSpec(3, 3)
    print("started statistics")
    x, y, z = make_grid(25, 50, 25)
    u_0 = generate_u(x, y, z)

    sl = slice(0, None, 50)
    true_future = true_future[sl]
    prediction = prediction[sl]

    true_u = []
    for a in true_future:
        true_u.append(calculate_velocities(x, y, z, a, u_0))
    true_u = np.array(true_u)

    predicted_u = []
    for a in prediction:
        predicted_u.append(calculate_velocities(x, y, z, a, u_0))
    predicted_u = np.array(predicted_u)

    print("started plotting")

    u_mean_true = np.average(true_u, axis=(0, 1, 3))
    u_mean_predicted = np.average(predicted_u, axis=(0, 1, 3))

    ux_mean_true = u_mean_true[:, X]
    ux_mean_predicted = u_mean_predicted[:, X]

    ux_square_mean_true = np.average(np.square(true_u - np.mean(true_u, axis=0)), axis=(0, 1, 3))[:, X]
    ux_square_mean_predicted = np.average(np.square(predicted_u - np.mean(predicted_u, axis=0)), axis=(0, 1, 3))[:, X]

    ux_third_mean_true = np.average(np.power(true_u, 3), axis=(0, 1, 3))[:, X]
    ux_third_mean_predicted = np.average(np.power(predicted_u, 3), axis=(0, 1, 3))[:, X]

    ux_fourth_mean_true = np.average(np.power(true_u, 4), axis=(0, 1, 3))[:, X]
    ux_fourth_mean_predicted = np.average(np.power(predicted_u, 4), axis=(0, 1, 3))[:, X]

    # # v_square_true = np.add(np.square(true_u[:,:,:,:,Y]), np.square(true_u[:,:,:,:,Z]))
    # v_square_true =
    # # v_square_predicted = np.add(np.square(predicted_u[:,:,:,:,Y]), np.square(predicted_u[:,:,:,:,Z]))
    # v_square_predicted =

    v_square_mean_true = np.average(np.square(true_u[:, :, :, :, Y]), axis=(0, 1, 3))
    v_square_mean_predicted = np.average(np.square(predicted_u[:, :, :, :, Y]), axis=(0, 1, 3))

    uv_mean_true = np.average(np.multiply(true_u[:, :, :, :, Y], true_u[:,:,:,:,X]), axis=(0, 1, 3))
    uv_mean_predicted = np.average(np.multiply(predicted_u[:, :, :, :, Y], predicted_u[:,:,:,:,X]), axis=(0, 1, 3))

    w_true = calculate_vorticity(x, y, z, true_u)
    w_pred = calculate_vorticity(x, y, z, predicted_u)

    wx_rms_true = np.std(w_true[:, :, :, :, X] - np.mean(w_true[:, :, :, :, X], axis=0), axis=(0, 1, 3))
    wy_rms_true = np.std(w_true[:, :, :, :, Y] - np.mean(w_true[:, :, :, :, Y], axis=0), axis=(0, 1, 3))
    wz_rms_true = np.std(w_true[:, :, :, :, Z] - np.mean(w_true[:, :, :, :, Z], axis=0), axis=(0, 1, 3))
    wx_rms_pred = np.std(w_pred[:, :, :, :, X] - np.mean(w_pred[:, :, :, :, X], axis=0), axis=(0, 1, 3))
    wy_rms_pred = np.std(w_pred[:, :, :, :, Y] - np.mean(w_pred[:, :, :, :, Y], axis=0), axis=(0, 1, 3))
    wz_rms_pred = np.std(w_pred[:, :, :, :, Z] - np.mean(w_pred[:, :, :, :, Z], axis=0), axis=(0, 1, 3))

    ax = plt.subplot(gs1[0])
    # ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set(xlabel=r"$\overline{u}$", ylabel='y')
    ax.plot(ux_mean_true, y, label='True profile')
    ax.plot(ux_mean_predicted, y, label='Predicted Profile')
    ax.legend(loc='upper left')

    ax = plt.subplot(gs1[1])
    # ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set(xlabel=r"$\overline{u^2}$", ylabel='y')
    ax.plot(ux_square_mean_true, y, label='True profile')
    ax.plot(ux_square_mean_predicted, y, label='Predicted Profile')
    ax.legend(loc='upper left')

    ax = plt.subplot(gs1[2])
    # ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set(xlabel=r"$\overline{uv}$", ylabel='y')
    ax.plot(uv_mean_true, y, label='True profile')
    ax.plot(uv_mean_predicted, y, label='Predicted Profile')
    ax.legend(loc='upper left')

    ax = plt.subplot(gs1[3])
    # ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set(xlabel=r"$\overline{v^2}$", ylabel='y')
    ax.plot(v_square_mean_true, y, label='True profile')
    ax.plot(v_square_mean_predicted, y, label='Predicted Profile')
    ax.legend(loc='upper left')

    ax = plt.subplot(gs1[4])
    # ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set(xlabel=r"$\overline{u^3}$", ylabel='y')
    ax.plot(ux_third_mean_true, y, label='True profile')
    ax.plot(ux_third_mean_predicted, y, label='Predicted Profile')
    ax.legend(loc='upper left')

    ax = plt.subplot(gs1[5])
    # ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set(xlabel=r"$\overline{u^4}$", ylabel='y')
    ax.plot(ux_fourth_mean_true, y, label='True profile')
    ax.plot(ux_fourth_mean_predicted, y, label='Predicted Profile')
    ax.legend(loc='upper left')

    ax = plt.subplot(gs1[6])
    # ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set(xlabel=r"$\omega_{x,rms}$", ylabel='y')
    ax.plot(wx_rms_true, y, label='True profile')
    ax.plot(wx_rms_pred, y, label='Predicted Profile')
    ax.legend(loc='upper left')

    ax = plt.subplot(gs1[7])
    # ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set(xlabel=r"$\omega_{y,rms}$", ylabel='y')
    ax.plot(wy_rms_true, y, label='True profile')
    ax.plot(wy_rms_pred, y, label='Predicted Profile')
    ax.legend(loc='upper left')

    ax = plt.subplot(gs1[8])
    # ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set(xlabel=r"$\omega_{z,rms}$", ylabel='y')
    ax.plot(wz_rms_true, y, label='True profile')
    ax.plot(wz_rms_pred, y, label='Predicted Profile')
    ax.legend(loc='upper left')

    plt.show()

def plot_dataset(fln):
    # train time 0 -> 52K
    # valid time 52K -> 68K
    # test time 68K -> 80K

    # train time 0 -> 208K
    # valid time 208K -> 272K
    # test time 272K -> 320K

    hf = h5py.File(fln, 'r')

    u = np.array(hf.get('/u'))
    t = np.array(hf.get('/t'))

    plt.figure(1)
    plt.subplot(511)
    plt.plot(t, u[:, 0])
    plt.subplot(512)
    plt.plot(t, u[:, 1])
    plt.subplot(513)
    plt.plot(t, u[:, 2])
    plt.subplot(514)
    plt.plot(t, u[:, 3])
    plt.subplot(515)
    plt.plot(t, u[:, 4])

    plt.figure(2)
    plt.subplot(511)
    plt.plot(t, u[:, 5])
    plt.subplot(512)
    plt.plot(t, u[:, 6])
    plt.subplot(513)
    plt.plot(t, u[:, 7])
    plt.subplot(514)
    plt.plot(t, u[:, 8])
    plt.show()

    plt.figure(3)
    from scipy.signal import find_peaks
    u0 = u[:, 0]
    peaks, properties = find_peaks(u0, prominence=0.3, width=100)
    plt.plot(u0)
    plt.plot(peaks, u0[peaks], "x")
    plt.vlines(x=peaks, ymin=u0[peaks] - properties["prominences"],
               ymax = u0[peaks], color = "C1")
    plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],
               xmax = properties["right_ips"], color = "C1")
    plt.show()
    print(peaks)

    plt.figure(4)
    from scipy.signal import find_peaks
    u0 = u[:, 0]
    peaks, properties = find_peaks(u0, prominence=0.5, width=100)
    plt.plot(u0)
    plt.plot(peaks, u0[peaks], "x")
    plt.vlines(x=peaks, ymin=u0[peaks] - properties["prominences"],
               ymax = u0[peaks], color = "C1")
    plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],
               xmax = properties["right_ips"], color = "C1")
    plt.show()
    print(peaks)

if __name__ == '__main__':
    plot_dataset('data/MFE.h5')
