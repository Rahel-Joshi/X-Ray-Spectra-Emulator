from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pickle
import emullver
import plotting
import _globals
from xspec import *
import torch
import torch.nn as nn
import torch.optim as optim
from xspec_comparison import *
from emullver import *
from error import *
import random
from emullver import params_to_code_units
import tensorflow as tf
from pathlib import Path

# mean of absolute log counts
SPECTRA_FACTOR = 2.720513452838532

def get_model1(weights_file=None):
    model = nn.Sequential(
            nn.Linear(5, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 4999))
    if weights_file:
        model.load_state_dict(torch.load(weights_file))
    return model

def get_model3(weights_file=None):
    model = nn.Sequential(
            nn.Linear(5, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Unflatten(1, (512, 1)),
            nn.Conv1d(in_channels=512, out_channels=128, kernel_size=3 , padding=1),
            nn.Upsample(scale_factor=2),
            nn.ReLU(),

            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2),
            nn.ReLU(),
            

            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2),
            nn.ReLU(),
            
            nn.Flatten(),
            nn.Linear(1024, 4999))
    if weights_file:
        model.load_state_dict(torch.load(weights_file))
    return model

def get_model2(weights_file=None):
    model = nn.Sequential(
            nn.Linear(5, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 4999))
    if weights_file:
        model.load_state_dict(torch.load(weights_file))
    return model

def get_data():
    '''
    return training, validation, and testing data from files
    '''
    train_x = np.load('model_data/train_x.npy')
    train_y = np.load('model_data/train_y.npy')
    valid_x = np.load('model_data/valid_x.npy')
    valid_y = np.load('model_data/valid_y.npy')
    test_x = np.load('model_data/test_x.npy')
    test_y = np.load('model_data/test_y.npy')

    train_X = torch.tensor(train_x, dtype=torch.float32)
    train_Y = torch.tensor(train_y, dtype=torch.float32)
    valid_X = torch.tensor(valid_x, dtype=torch.float32)
    valid_Y = torch.tensor(valid_y, dtype=torch.float32)
    test_X = torch.tensor(test_x, dtype=torch.float32)
    test_Y = torch.tensor(test_y, dtype=torch.float32)

    return train_X, train_Y, valid_X, valid_Y, test_X, test_Y

def model_get_errors(model, X, Y):
    '''
    return array of tuples of (gamma, Afe, logxi, Ecut, Incl, error)
    '''
    errors = []
    count = 0
    for X, Y in zip(test_X, test_Y):
        pred = model(X)
        error = torch.mean(torch.square(pred - Y))
        errors.append((*X, error.item()))
        count += 1
        if count % 1000 == 0:
            print(count)
    errors = np.array(errors)
    return errors

def create_model_errors(model, model_name):
    '''
    creates file of array of (gamma, Afe, logxi, Ecut, Incl, error)
    '''

    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = get_data()

    train_Y = train_Y / SPECTRA_FACTOR
    valid_Y = valid_Y / SPECTRA_FACTOR
    test_Y = test_Y / SPECTRA_FACTOR

    train_errors = model_get_errors(model, train_X, train_Y)
    np.save('model_errors/' + model_name + '_train_errors.npy', train_errors)

    valid_errors = model_get_errors(model, valid_X, valid_Y)
    np.save('model_errors/' + model_name + '_valid_errors.npy', valid_errors)

    test_errors = model_get_errors(model, test_X, test_Y)
    np.save('model_errors/' + model_name + '_valid_errors.npy', test_errors)


class EncoderDecoderModel(nn.Module):
    def __init__(self, weights_file=None):
        super(EncoderDecoderModel, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(4999, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4999),
        )

        if weights_file:
            self.load_state_dict(torch.load(weights_file))
    
    def forward(self, x):
        # Pass through encoder
        encoded = self.encoder(x)
        # Pass through decoder
        decoded = self.decoder(encoded)
        return decoded


def train_model(model, total_steps, batch_size, lr, model_name, train_X, train_Y, valid_X, valid_Y):
    '''
    train model and collect weights and loss data
    '''
    if Path('model_weights/' + model_name + '.pth').is_file():
        return

    # train_X, train_Y, valid_X, valid_Y, test_X, test_Y = get_data()

    # train_Y = train_Y / SPECTRA_FACTOR
    # valid_Y = valid_Y / SPECTRA_FACTOR
    # test_Y = test_Y / SPECTRA_FACTOR

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    steps_train_loss = []
    steps_valid_loss = []
    best_step = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=8)

    for step in range(total_steps):
        for i in range(0, len(train_X), batch_size):
            X_batch = train_X[i:i + batch_size]
            Y_pred = model(X_batch)
            Y_batch = train_Y[i:i + batch_size]
            loss = loss_fn(Y_pred, Y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step(loss)
        print(f'Finished step {step}, latest loss {loss}')
        steps_train_loss.append(loss.item())

        for param_group in optimizer.param_groups:
            print('learning rate: ' + str(param_group['lr']))
        
        valid_Y_pred = model(valid_X)
        valid_loss = loss_fn(valid_Y_pred, valid_Y)
        steps_valid_loss.append(valid_loss.item())
        print(f'Validation score {valid_loss}')

        if valid_loss < steps_valid_loss[best_step]:
            torch.save(model.state_dict(), 'model_weights/' + model_name + '.pth')
            best_step = step
        
        print(f'Best step {best_step}')

        np.save('model_loss/' + model_name + '_train_loss.npy', steps_train_loss)
        np.save('model_loss/' + model_name + '_valid_loss.npy', steps_valid_loss)

def model_get_tuples(model, X, Y):
    """
    return (gamma, Afe, logxi, Ecut, i, error, index) tuples for model
    """
    tuples = []
    for i, (x, y) in enumerate(zip(X, Y)):
        pred = model(x)
        pred = 2.720513452838532 * pred
        error = torch.mean(torch.square(pred - y)).item()
        x = params_to_physical_units(x)
        tuples.append(np.array([*x, error, i]))
    
    tuples = np.array(tuples)
    return tuples

def model_plots(model_list, tuples, Y, num):
    """
    plot emulated spectra
    """
    for i in range(num):
        # set up variables
        gamma, Afe, logxi, Ecut, Incl, error, index = tuples[i]
        x = [gamma, Afe, logxi, Ecut, Incl]
        x = params_to_code_units(x)
        y = Y[int(index)]
        with torch.no_grad():
            # the inputs and graph setup
            x = torch.tensor(np.array(x), dtype=torch.float32)
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(2, 1, 1)
            bins = get_energy_bins()
            
            res_list = []
            
            # add emulated spectra
            for j, model in enumerate(model_list):
                pred = model(x)
                pred = 2.720513452838532 * pred
                res = y - pred
                res_list.append(res)
                ax.plot(bins, pred, label='model ' + str(j+1))
        
            # text
            ax.plot(bins, y, label='table')
            ax.set_xscale('log')
            ax.set_xlabel('Energy [keV]')
            ax.set_ylabel(r"$\log(\rm photons / cm^2 / s / keV)$")
            pars = (
                r"$\Gamma=%.2f$" % gamma
                + "\n"
                + r"$\rm A_{Fe}=%.2f$" % Afe
                + "\n"
                + r"$\log(\xi)=%.2f$" % logxi
                + "\n"
                + r"$E_{\rm cut}=%.2f$" % Ecut
                + "\n"
                + r"$i=%.2f$" % Incl
            )
            
            ax.text(0.1, 0.1, pars, transform=ax.transAxes, fontsize=10)

            plt.legend()
            
            # plot residuals
            ax = fig.add_subplot(2, 1, 2)
            for j, res in enumerate(res_list):
                ax.plot(bins, res, label='model ' + str(j+1), alpha=.5)
            plt.axhline(y=0, color='r', linestyle='-')
            ax.set_xscale('log')
            ax.set_xlabel('Energy [keV]')
            ax.set_ylabel('Residual')

            plt.legend()
            plt.show()
   
def model_plot_loss(valid_loss_file, train_loss_file, name='Model'):
    """
    plot model's validation and training loss
    """
    valid_loss = np.load(valid_loss_file)
    train_loss = np.load(train_loss_file)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_loss, label='training loss', alpha=.7)
    ax.plot(valid_loss, label='validation loss', alpha=.7)
    fig.legend()
    plt.title(name)
    ax.set_yscale('log')
    ax.set_xlabel('Steps')
    ax.set_ylabel('log(MSE Loss)')