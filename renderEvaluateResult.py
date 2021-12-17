import pickle
from matplotlib import pyplot as plt

with open('evaluate/result.pkl', 'rb') as evaluateResultFile:
    result = pickle.load(evaluateResultFile)
    
    # print("result: ", result)
    data_window_size = 12
    firstPrice = result['prices'][data_window_size]
    total_value_history = [0 for x in range(data_window_size)]
    total_value_history.extend([(lambda v: v*firstPrice)(v) for v in result['total_value_history'] ])

    plt.figure(figsize=(15,6))
    plt.cla()
    plt.grid(True)
    plt.plot(result['prices'])
    plt.plot(total_value_history)
    plt.plot(result['short_ticks'], result['prices'][result['short_ticks']], 'ro')
    plt.plot(result['long_ticks'], result['prices'][result['long_ticks']], 'go')
    # plt.plot(result['hold_ticks'], result['prices'][result['hold_ticks']], 'yo')
    plt.suptitle(
            "Total Reward: %.6f" % result['total_reward'] + ' ~ ' +
            "Latest value: %.6f" % total_value_history[-1]
        )
    plt.show()
