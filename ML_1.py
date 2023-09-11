import json
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
import os
from numpy import savetxt

root = tk.Tk()
root.withdraw()


# Read spectrum JSON File
def getFile(src, dist, time, num):
    file_path = (
        os.getcwd()
        + "/data/"
        + src
        + "/1K/"
        + "spect-"
        + src
        + "-1K-"
        + str(time)
        + "sec-"
        + str(dist)
        + "cm-"
        + str(num)
        + ".json"
    )  # filedialog.askopenfilename()
    if file_path != "":
        with open(file_path, "r") as f:
            data = json.load(f)
            histogram = data["histogram"]
    else:
        file_path = ""

    return histogram


# Normalize data spectrum
def norm(histo):
    Peak = max(histo)
    for i in range(len(histo)):
        histo[i] = histo[i] * 255 / Peak

    return histo


# Background Divide
def bckDiv(histo, avgBck):
    for i in range(len(histo)):
        if avgBck[i] > 0:
            histo[i] /= avgBck[i]
        else:
            histo[i] = 0

    return histo


# Background Substraction
def bckSubs(histo, avgBck):
    for i in range(len(histo)):
        histo[i] = histo[i] - avgBck[i]
        if histo[i] < 0:
            histo[i] = 0

    return histo


# Get Feature Extraction, 32x32 pixels data
def getFt(histo, srcname, dist, tim, num):
    n = 0
    spect = [[0 for col in range(32)] for row in range(32)]
    for x in range(len(histo)):
        a = x % 32
        spect[n][a] = histo[x]
        if x > 0 and a == 0:
            n += 1

    file_path = os.getcwd() + "/data/" + srcname + "/Ft_Image/"

    # np.savetxt(file_path + srcname + "-" + str(dist) + "cm-" + str(tim) + "sec-" + str(num) + ".csv",
    #      spect,
    #      delimiter =", ",
    #      fmt ='% s')

    # savetext()
    # im = plt.imshow(spect, cmap='RdYlBu_r', interpolation='nearest')
    # plt.colorbar(im)
    # plt.figure(num)
    # plt.tight_layout()
    fig = plt.figure(frameon=False)
    fig.set_size_inches(3, 3)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(spect, aspect="auto")
    plt.savefig(
        file_path
        + srcname
        + "-"
        + str(dist)
        + "cm-"
        + str(tim)
        + "sec-"
        + str(num)
        + ".png"
    )
    print(srcname + "-" + str(dist) + "cm-" + str(tim) + "sec-" + str(num))
    # plt.show()

    return spect


# Background Substraction
def getBck():
    histo = []
    avgHis = []

    path = os.getcwd() + "/data/Background/1K"
    json_file_names = [
        filename for filename in os.listdir(path) if filename.endswith(".json")
    ]

    for json_file_name in json_file_names:
        with open(os.path.join(path, json_file_name)) as json_file:
            json_text = json.load(json_file)
            json_int = list(map(int, json_text["histogram"]))
            histo.append(json_int)

    resHis = np.sum(histo, 0)
    for k in range(len(resHis)):
        a = resHis[k] / len(histo)
        avgHis.append(a)

    Peak = max(avgHis)
    avgHis /= Peak / 255

    return avgHis


def main():
    src = "Cs-134"
    time = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    dist = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

    for n in time:
        for m in dist:
            for s in range(50):
                avgBck = getBck()
                histo = getFile(src, m, n, s)
                histo = norm(histo)
                histo = bckDiv(histo, avgBck)
                # if n == 60 :
                # plt.figure(s)
                # plt.plot(histo)
                # plt.show()

                histo = norm(histo)
                getFt(histo, src, m, n, s)


main()
