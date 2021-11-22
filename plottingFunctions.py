import matplotlib.pyplot as plt

def plotPs(x_range, n_runs, runs, title="Gap in Profit"):
    plt.title("Profit Gap vs N")
    # plt.plot(full_results[:,0], full_results[:,2], label="Apply guess vector", color="green")
    for j in range(n_runs):
        plt.plot(x_range, runs[:,j], label="Trial " + str(j + 1))
    plt.xlabel("Prob")
    plt.ylabel(title)
    plt.legend()
    plt.show()
