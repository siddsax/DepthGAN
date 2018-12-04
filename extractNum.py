import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt



fl = open('Errors_sci_1.txt').read().split('\n')
# print(fl)

losses = []
for i, line in enumerate(fl):
    if (i%7==1):
        err = line.split()
        loss = []
        for j, itm in enumerate(err):
            if j%2:
                loss.append(float(itm))
        losses.append(np.array(loss))

Origlosses = np.array(losses)
print(Origlosses.shape)

losses = []
for i, line in enumerate(fl):
    if (i%7==2):
        err = line.split()
        loss = []
        for j, itm in enumerate(err):
            if j%3==2 and j > 2:
                loss.append(float(itm))
        losses.append(np.array(loss))

d1 = np.array(losses)
print(d1.shape)

losses = []
for i, line in enumerate(fl):
    if (i%7==4):
        err = line.split()
        loss = []
        for j, itm in enumerate(err):
            if j%3==2 and j > 2:
                loss.append(float(itm))
        losses.append(np.array(loss))

d2 = np.array(losses)

d1_o = d1
d1 = d1[~np.isinf(d1_o).any(axis=1)]
d2 = d2[~np.isinf(d1_o).any(axis=1)]
Origlosses = Origlosses[~np.isinf(d1_o).any(axis=1)]

x = np.arange(d2.shape[0])
plt.plot(x, Origlosses[:,0],label="orig")
plt.plot(x, d1[:,0],label="d1")
plt.plot(x, d2[:,0],label="d2")

# Origlosses[:,0].reshape((63,1)),
names = ['rmse', 'rel', 't1', 't2', 't3']
toPrint = ""
numF = d1.shape[0]
for i in range(5):
    if i < 2:
        RMSEmn = np.amin(np.concatenate((Origlosses[:,i].reshape((numF,1)), d1[:,i].reshape((numF,1)), d2[:,i].reshape((numF,1))), axis=1), axis=1).mean()
        RMSEmnwO = np.amin(np.concatenate((d1[:,i].reshape((numF,1)), d2[:,i].reshape((numF,1))), axis=1), axis=1).mean()
    else:
        RMSEmn = np.amax(np.concatenate((Origlosses[:,i].reshape((numF,1)), d1[:,i].reshape((numF,1)), d2[:,i].reshape((numF,1))), axis=1), axis=1).mean()
        RMSEmnwO = np.amax(np.concatenate((d1[:,i].reshape((numF,1)), d2[:,i].reshape((numF,1))), axis=1), axis=1).mean()
    Org = Origlosses[:,i].mean()
    toPrint += "{} Orgl {} wtO {} wO {}\n".format(names[i], Org, RMSEmnwO, RMSEmn)
print(toPrint)
# print(RMSEmnwO)
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
# plt.show()
