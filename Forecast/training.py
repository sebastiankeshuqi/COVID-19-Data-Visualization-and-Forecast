'''
Reference: https://colab.research.google.com/drive/1nQYJq1f7f4R0yeZOzQ9rBKgk00AfLoS0#scrollTo=3_15jwwrASTP&forceEdit=true&sandboxMode=true
           https://www.curiousily.com/posts/time-series-forecasting-with-lstm-for-daily-coronavirus-cases/
'''
import torch
import numpy
import pandas
import seaborn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Build the forecast model


class COVIDPredictor(torch.nn.Module):

    def __init__(self, input_num, hidden_layer_num, recurrent_layer_num, batch_size):
        '''
        :paras:recurrent_layer_num – Number of recurrent layers
        '''
        super(COVIDPredictor, self).__init__()
        self.batch_size = batch_size
        self.input_num = input_num
        self.hidden_layer_num = hidden_layer_num
        self.recurrent_layer_num = recurrent_layer_num
        self.lstm = torch.nn.LSTM(
            input_size=self.input_num,
            hidden_size=self.hidden_layer_num,
            num_layers=self.recurrent_layer_num,
            dropout=0.6
        )
        self.linear = torch.nn.Linear(
            in_features=hidden_layer_num, out_features=1)

    def reset_hidden_state(self):  # Use a stateless LSTM
        self.hidden = (
            torch.zeros(self.input_num, self.batch_size,
                        self.hidden_layer_num),
            torch.zeros(self.input_num, self.batch_size, self.hidden_layer_num)
        )
    def forward(self, inputs):  # Pass the data through the LSTM
        lstm_out, self.hidden = self.lstm(
            inputs.view(len(inputs), self.batch_size, -1),
            self.hidden
        )
        last_time_step = lstm_out.view(
            self.batch_size, len(inputs), self.hidden_layer_num)[-1]
        y_predict = self.linear(last_time_step)
        return y_predict


def preprocessing():
    data_file = pandas.read_csv('time_series_covid19_confirmed_global.csv')
    data_file = data_file.iloc[:, 4:]
    daily_cases = data_file.sum()

    days = daily_cases.shape[0]
    test_size = int(days * 0.2)  # Use 20% data for tests

    train_data = daily_cases[:-test_size]
    test_data = daily_cases[-test_size:]

    '''
    Use MinMaxScaler to speed up training and get better performance
    '''

    scaler = MinMaxScaler()
    scaler = scaler.fit(numpy.expand_dims(train_data, axis=1))
    train_data = scaler.transform(numpy.expand_dims(train_data, axis=1))
    test_data = scaler.transform(numpy.expand_dims(test_data, axis=1))

    batch_size = 7
    X_train, y_train = create_sequences(train_data, batch_size)
    X_test, y_test = create_sequences(test_data, batch_size)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()

    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    model = COVIDPredictor(
        input_num=1,
        hidden_layer_num=512,
        recurrent_layer_num=1,
        batch_size=batch_size
    )
    model, train_hist, test_hist = train(
        model,
        X_train,
        y_train,
        X_test,
        y_test
    )

    plt.plot(train_hist, label="Training loss")
    plt.plot(test_hist, label="Test loss")
    plt.ylim((0, 5))
    plt.show()


def train(model, X_train, y_train, test_data=None, y_test=None):
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    epoch_num = 60

    train_hist = numpy.zeros(epoch_num)
    test_hist = numpy.zeros(epoch_num)

    for i in range(epoch_num):
        model.reset_hidden_state()

        y_pred = model(X_train)

        loss = loss_fn(y_pred.float(), y_train)

        if test_data is not None:
            with torch.no_grad():
                y_test_pred = model(test_data)
                test_loss = loss_fn(y_test_pred.float(), y_test)
            test_hist[i] = test_loss.item()

            if i % 10 == 0:
                print(
                    f'Epoch {i} train loss: {loss.item()} test loss: {test_loss.item()}')
        elif i % 10 == 0:
            print(f'Epoch {i} train loss: {loss.item()}')

        train_hist[i] = loss.item()

        optimiser.zero_grad()

        loss.backward()

        optimiser.step()

    return model.eval(), train_hist, test_hist


def main():
    data_file = pandas.read_csv('time_series_covid19_confirmed_global.csv')
    data_file = data_file.iloc[:, 4:]
    daily_cases = data_file.sum()

    scaler = MinMaxScaler()
    scaler = scaler.fit(numpy.expand_dims(daily_cases, axis=1))
    all_data = scaler.transform(numpy.expand_dims(daily_cases, axis=1))

    batch_size = 7

    X_all, y_all = create_sequences(all_data, batch_size)
    X_all = torch.from_numpy(X_all).float()
    y_all = torch.from_numpy(y_all).float()
    model = COVIDPredictor(
        input_num=1,
        hidden_layer_num=512,
        recurrent_layer_num = 1,
        batch_size=batch_size
    )
    model, train_hist, _ = train(model, X_all, y_all)
    predict_days = 12
    with torch.no_grad():
        test_seq = X_all[:1]
        preds = []

        for _ in range(predict_days):
            y_test_pred = model(test_seq)
            pred = torch.flatten(y_test_pred).item()
            preds.append(pred)
            new_seq = test_seq.numpy().flatten()
            new_seq = numpy.append(new_seq, [pred])
            new_seq = new_seq[1:]
            test_seq = torch.as_tensor(new_seq).view(1, batch_size, 1).float()

        predicted_cases = scaler.inverse_transform(
            numpy.expand_dims(preds, axis=0)
        ).flatten()

        plt.plot(daily_cases, label='Historical Daily Cases')
        plt.plot(predicted_cases, label='Predicted Daily Cases')
        plt.show()


def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return numpy.array(xs), numpy.array(ys)


if __name__ == '__main__':
    # preprocessing()
    main()
