import json
import torch
import pandas as pd
import os


class TrainNN:
    def __init__(self, model, exp_dir, x_train, y_train, x_val, y_val):
        self.model = model
        self.exp_dir = exp_dir
        self.num_training_inst = x_train.shape[0]

        self.x_train = torch.tensor(x_train, device=self.model.device, dtype=self.model.dtype)
        self.y_train = torch.tensor(y_train, device=self.model.device, dtype=torch.long).squeeze()
        self.x_val = torch.tensor(x_val, device=self.model.device, dtype=self.model.dtype)
        self.y_val = torch.tensor(y_val, device=self.model.device, dtype=torch.long).squeeze()

        # define the name of the directory to be created
        if not os.path.isdir(self.exp_dir):
            os.makedirs(self.exp_dir)

    def run_train(self, n_epochs, lr=0.001, batch_size=64, weight_decay=0):
        # set model to training-mode
        self.model.train()

        # Loss and optimizer
        criterion = torch.nn.CrossEntropyLoss() #torch.nn.AbsCriterion()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Train
        loss_hist_train = []
        loss_hist_val = []
        n_epochs_hist = []

        for i in range(n_epochs):

            for batch in range(0, int(self.num_training_inst / batch_size)):
                # Prepare Batch
                batch_x = self.x_train[batch * batch_size: (batch + 1) * batch_size, :]
                batch_y = self.y_train[batch * batch_size: (batch + 1) * batch_size]

                # foward step
                outputs = self.model(batch_x)

                # calculate loss
                loss = criterion(outputs, batch_y)

                # backward step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Calculate loss and store
            if i % 10 == 0:
                # set model to evaluation
                self.model.eval()

                outputs = self.model(self.x_train)
                loss = criterion(outputs, self.y_train)
                loss_hist_train.append(loss.item())

                outputs_val = self.model(self.x_val)
                loss_val = criterion(outputs_val, self.y_val)
                loss_hist_val.append(loss_val.item())

                self.model.train()
                n_epochs_hist.append(i)

                print(i, ' train_loss: ', loss.item(), 'validate_loss: ', loss_val.item())


        # Save the model
        torch.save(self.model, self.exp_dir + self.model.name + '.pt')

        # Save temporal course of loss (training-data / validation-data)
        pd.DataFrame(loss_hist_train).to_csv(self.exp_dir + '/errors_train__' + self.model.name + '__.csv', index=False)
        pd.DataFrame(loss_hist_val).to_csv(self.exp_dir + '/errors_val__' + self.model.name + '__.csv', index=False)

        # Store meta data
        meta_data = {'n_epochs': n_epochs, 'lr': lr, 'batch_size': batch_size}
        with open(self.exp_dir + '/data.json', 'w') as fp:
            json.dump(meta_data, fp)

        return
