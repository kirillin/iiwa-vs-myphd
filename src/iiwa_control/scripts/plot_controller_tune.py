import numpy as np


# f = open("log_control.txt", r)

import matplotlib.pyplot as plt 


# tau_fb.transpose() << " " << q.transpose() << " " << dq.transpose() << " " << (q_des - q).transpose() << " " << t << "\n";
# a = np.loadtxt("log_control.txt")
a = np.genfromtxt("log_control.txt", skip_footer=5);

t = a[:,-1]
t[0] = 0

tau_fb = a[:, 0:6]
q = a[:, 7:13]
dq = a[:, 14:20]
err = a[:, 21:27]


print(a.shape)
print(tau_fb.shape)

plt.subplot(411)
plt.plot(t, tau_fb, linewidth=1)
plt.xlim(0.00, np.max(t))
plt.legend(['1', '2', '3', '4', '5', '6', '7'])
plt.ylabel('$\tau_{cmd}$, Nm')

plt.subplot(412)
plt.plot(t, q)
plt.xlim(0.00, np.max(t))
plt.ylabel('$q^{*} - q, rad$')

plt.subplot(413)
plt.plot(t, dq)
plt.xlim(0.00, np.max(t))
plt.ylabel('$\dot q, rad/s$')

plt.subplot(414)
plt.plot(t, err)
plt.xlim(0.00, np.max(t))
plt.ylabel('$\Delta q, rad$')
plt.xlabel('$t$')


plt.show()