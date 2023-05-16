# import numpy
# import matplotlib.pyplot as plt
# x = numpy.arange(0, 1, 0.05)
# y = numpy.power(x, 2)

# fig = plt.figure()
# ax = fig.gca()
# ax.set_xticks(numpy.arange(0, 1, 0.1))
# ax.set_yticks(numpy.arange(0, 1., 0.1))
# plt.plot(x, y)
# plt.grid()
# plt.show()

# from matplotlib import pyplot as plt

# fig = plt.figure()
# ax = fig.add_subplot(111)

# avg = [0.0538,0.0413,0.0731,0.1778,0.8764,3.4848,6.9722,48.4756,160.9765]
# navg = range(2,11)

# line,  = plt.plot(navg,avg)
# for xy in zip(navg, avg):                                       # <--
#     ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data') # <--

# line.set_label('Average time over 30 instances')

# plt.legend()
# plt.grid()
# plt.show()


# worst = [0.0660,0.0600,1.1160,1.3169,4.8213,9.3613,228.6678,2214.79,58983.394]

# nworst = [2,3,4,5,6,7,8,9,11]

# line1,  = plt.plot(nworst,worst)
# for xy in zip(nworst, worst):                                       # <--
#     ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data') # <--

# line1.set_label('worst time')
# plt.legend()
# plt.grid()
# plt.show()


