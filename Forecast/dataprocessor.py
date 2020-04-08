'''
Reference: https://colab.research.google.com/drive/1nQYJq1f7f4R0yeZOzQ9rBKgk00AfLoS0#scrollTo=3_15jwwrASTP&forceEdit=true&sandboxMode=true
'''
import os
import csv
import wget
import pandas
import requests
import matplotlib.pyplot as plt

# Check this website[https://data.humdata.org/dataset/novel-coronavirus-2019-ncov-cases] to find the url to download the updated data
data_url = 'https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_confirmed_global.csv&filename=time_series_covid19_confirmed_global.csv'


def download_data1():
    '''
    This download sentense works just smoothly on Windows.
    But if you are running this on OSX system, it probably pops mistake like
        "urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1108)>"
    So you should open Your Macintosh Folder/Applications/Python3.X
    Open the two files:
        Install Certificates.command
        Update Shell Profile.command
    Then everything will just work well.
    '''
    data_file = pandas.read_csv(data_url)

    return data_file


def download_data2():
    '''
    This download sentense works just smoothly on Windows.
    But if you are running this on OSX system, it probably pops mistake like
        "urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1108)>"
    So you should open Your Macintosh Folder/Applications/Python3.X
    Open the two files:
        Install Certificates.command
        Update Shell Profile.command
    Then everything will just work well.
    '''

    filename = wget.download(data_url)

    data_file = pandas.read_csv(filename)

    return data_file


def visualize(data_file):
    '''
    Use this function to visualize the daily confirmed cases
    '''
    data_file = data_file.iloc[:, 75:]

    daily_cases = data_file.sum(axis=0)

    plt.plot(daily_cases)
    plt.title('Cumulative daily cases')
    plt.show()
