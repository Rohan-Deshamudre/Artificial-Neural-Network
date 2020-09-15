import matplotlib.pyplot as plt
import os.path


def get_values(pathname):
    path = pathname
    f = open(path, "r")
    names = []
    values = []
    line = f.readline().split(",")
    for i in range(8):
        values.append(float(line[i]))

    f.close()
    return values

def values_multiple_files(pathname):
    path = os.path.abspath(__file__ + ".\\..\\..\\resources\\"+pathname)
    f = open(path, "r")
    it = 0
    success = 0
    for i in range(10):
        x = f.readline().split(",")
        it += int(x[0])
        success += float(x[1])
    it = it / 10
    success = success / 10
    f.close()
    return it, success


if __name__ == "__main__":
    path1 = os.path.abspath(__file__ + ".\\..\\..\\resources\\30.txt")
    path2 = os.path.abspath(__file__ + ".\\..\\..\\resources\\validation_set.txt")
    # path3 = os.path.abspath(__file__ + "..\\..\\..\\..\\resources\\gamma_07.txt")

    paths = ["7.txt", "10.txt", "16.txt", "20.txt", "30.txt"]

    names = ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "t"]
    values = get_values(path1)

    # names, values = get_values(path1)
    #fig, ax1 = plt.subplots()
    #color = 'tab:blue'
    plt.bar(names, values)
    # ax1.plot(names, values[1], marker='o', label="2nd validation set")
    # ax1.plot(names, values[2], marker='o', label="3rd validation set")
    # ax1.plot(names, values[3], marker='o', label="4th validation set")
    # ax1.plot(names, values[4], marker='o', label="5th validation set")
    # ax1.plot(names, values[5], marker='o', label="6th validation set")
    # ax1.plot(names, values[6], marker='o', label="7th validation set")
    # ax1.plot(names, values[7], marker='o', label="test set")
    # ax1.tick_params(axis='y', labelcolor=color)
    #
    # ax2 = ax1.twinx()
    # color = 'tab:red'
    # ax2.plot(names, values_success, marker='o', label="Properly assigned elements of test set", color=color)
    # ax2.tick_params(axis='y', labelcolor=color)

    plt.suptitle("Train and validation sets performace after training")
    #plt.ylabel("avergae steps per trial")
    plt.axis([-0.5, 7.5, min(values)*0.99, max(values)*1.01])
    # ax2.axis([-0.5, 10.5, min(values_success)*0.8, max(values_success)*1.05])
    #ax1.legend(loc='best')
    #ax2.legend(loc='upper left')

    plt.savefig(os.path.abspath(__file__ + ".\\..\\..\\resources\\validation_test.png"))
