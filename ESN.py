import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from ipywidgets import *
from IPython.display import *
import sys


# ------------------------------------------- #
# ---------------- VARIABLES ---------------- #
# ------------------------------------------- #


# Number of trainings and tests
train_time = 40000000
test_time = 10000

# Number of reservoir units
N = 50

# Number of inputs
K = 1

# Number of outputs
L = 1

# Generate the weight restraint in Wout
resWout = 1e-4

# Generate the reservoir weights matrix
spectral_radius = 1
W = np.random.uniform(-1, 1, (N, N))
radius = np.max(np.abs(np.linalg.eig(W)[0]))
W *= spectral_radius / radius

# Generate the input weights matrix
sc = 1
Win = np.random.uniform(-sc, sc, (N, K))

# Generate the output feedback reservoir weights matrix
sc2 = 1
Wfb = np.random.uniform(-sc2, sc2, (N, K))


# ------------------------------------------- #
# ---------------- FUNCTIONS ---------------- #
# ------------------------------------------- #


def set_seed(seed=None):
    """
    Create the seed (for random values) variable if None
    """

    if seed is None:
        import time
        seed = int((time.time()*10**6) % 4294967295)
    try:
        np.random.seed(seed)
        print("Seed:", seed)
    except:
        print("!!! WARNING !!!: Seed was not set correctly.")
    return seed


def plot_figure(data, title):
    """
    Plot the figure of the given data using MatPlotLib

    Args:
        data (array of float): Dataset we want to plot
        title (string): String to print as the title
    """

    f = plt.figure()
    ax = f.add_subplot(1,1,1)
    ax.set_xlabel('Time')
    f.suptitle(title)
    ax.plot(data)


def random_data(low, high, size):
    """
    Generate a random dataset for training and testing purposes

    Args:
        loc (float): mean
        scale (float): standard deviation
        size (int or tuple of ints):

    Returns:
        The dataset, in a form of a size x dim array
    """

    dataset = np.random.uniform(low, high, size)
    return dataset


def state_update(W, x_n, Win, u_n1, Wfb, y_n):
    """
    Compute the x_n+1 reservoir state

    Args:
        W (N x N matrix): reservoir weight matrix
        x_n (N vector): Previous reservoir state
        Win (N x K matrix): Input weight matrix
        u_n1 (K vector): Input signal
        Wfb (N x L matrix): Output feedback reservoir weight matrix
        y_n (L vector): Output feedback signal

    Returns:
        N-dimensional array of float: x_n+1
    """
    #print("Shapes:", W.shape, x_n.shape, Win.shape, u_n1.shape)
    return np.tanh(np.dot(W, x_n) + np.dot(Win, u_n1) + np.dot(Wfb, y_n))


def get_Wout(X, Y):
    """
    Generate the ouput weights matrix

    Args:
        X (N x T matrix): Predictions matrix
        Y (K x T matrix): Target output

    Returns:
        K x N matrix: Output weights
    """

    #Wout = np.dot(Y, np.linalg.pinv(X))
    Wout = np.dot(Y, np.dot(X.T, np.linalg.inv(np.dot(X, X.T) + resWout * np.identity(X.shape[0]))))
    return Wout


def train(input, Y, T):
    """
    Trains the reservoir with training data for N times

    Args:
        input (size x K matrix):
        Y (L vector):
        T (int): Training time

    Returns:
        N x T matrix: Output weights matrix
    """

    # Compute a random x_n at first
    x_n = np.zeros(N)
    # Will store all values of x_n during time T
    X = np.empty((N, T))

    for i in range(1, T):

        # Compute the input u_n
        u_n = input[:,i]

        # Compute the target y_n
        y_n = Y[:,i-1]

        # Add x_n to X
        X[:,i-1] = x_n

        # Compute x_n+1
        x_n = state_update(W, x_n, Win, u_n, Wfb, y_n)

    Wout = get_Wout(X, Y)

    return Wout


def predict(input, Wout, Y, T):
    """
    Test the output weights on a new dataset

    Args:
        input (size x K matrix): Testing dataset
        Wout (K x N matrix): Output weights
        Y (L vector): Target output

    """

    costs = np.empty(T) # Store all error costs
    predictions = np.empty(T)
    targets = np.empty(T)

    # New x_n
    x_n = np.random.uniform(-1, 1, N)

    for i in range(T):

        u_n = input[:, i]

        # Compute the target y_n
        y_n = np.dot(Wout, x_n)

        # Compute the cost
        prediction = np.dot(Wout, x_n)
        target = Y[:, i-1]

        costs[i] = np.square(np.linalg.norm(prediction - target))

        predictions[i] = prediction

        targets[i] = target

        # Compute x_n+1
        x_n = state_update(W, x_n, Win, u_n, Wfb, y_n)

    plot_figure(targets, "Target outputs")
    plot_figure(predictions, "ESN predictions")
    plot_figure(costs, "Errors")
    plt.show()

    cost = np.sum(costs)
    cost = np.sqrt(cost / T)


    print("Error: {:.1e}".format(cost))


# ------------------------------------------- #
# ----------------- TESTING ----------------- #
# ------------------------------------------- #


# Initiate the seed value
seed = set_seed()

# Create training dataset
data_train = random_data(-1, 1, (K, train_time))
#data_train = np.sin(np.arange(train_time)/100)[None, :]
Y_train = 1 * data_train # Target values for training

# Create testing dataset
data_test = random_data(-1, 1, (K, test_time))
#data_test = np.sin(np.arange(test_time)/100)[None, :]
Y_test = 1 * data_test # Target values for testing

# Learning step
Wout = train(data_train, Y_train, train_time)

# Predictive step
predict(data_test, Wout, Y_test, test_time)
