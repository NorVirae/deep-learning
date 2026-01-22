import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation

from matplotlib.animation import PillowWriter

plt.style.use(["default"])

x = np.linspace(0, 15, 30)
y = np.sin(x) + 0.1 * np.random.randn(len(x))

x2 = np.linspace(0, 15, 100)
y2 = np.sin(x2)

# plt.figure(figsize=(8,3))
# plt.xlabel("Time [s]")
# plt.ylabel("Voltage [V]")


# plt.plot(x, y, "o--", color="red", lw=0.4, ms=4, label="Component 1")
# plt.plot(x2, y2, "-", color="green", lw=0.4, ms=4, label="Component 2")

# plt.xlim(right=6) # basically used to number the axis in question to any direction shown,  there by giving space.
# plt.legend(loc="upper right", fontsize=10, ncol=5)


# # Plot a histogram

# res = np.random.randn(1000)*0.2 + 0.4
# res2 = np.random.randn(1000)*0.2 + 0.4

# print(res)
# plt.hist(res, bins=30, density=True, histtype="step")
# plt.hist(res2, bins=30, density=True, histtype="step")

# # doing multiple plots

# fig, axes = plt.subplots(2, 2, figsize=(8, 3))

# print(axes)

# ax = axes[0][0]
# ax.tick_params(axis="both", labelsize=4)
# ax.plot(x, y)
# ax.text(1, 0.1, "Bad guy", transform=ax.transAxes)

# ax = axes[0][1]
# ax.plot(x2, y2)

# ax = axes[1][0]
# ax.hist(res, bins=30, density=True, histtype="step")

# ax = axes[1][1]
# ax.hist(res2, bins=30, density=True, histtype="step")

# fig.text(0.5, 0.04, "$\Delta E$", ha="left", fontsize=20)


# contour plots

_ = np.linspace(-1, 1, 100)
x, y = np.meshgrid(_, _)
z = x**2 + x * y

# plt.contourf(x, y, z, levels=30, vmax=1.68, cmap="plasma" )
# plt.colorbar(label="hey")


# Line contour plots
# cs  = plt.contour(x, y, z)
# plt.clabel(cs, fontsize=30)


# Make a 3d plot
# fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
# ax.plot_surface(x, y, z)


# w = 3
# _ = np.linspace(-3, 3, 100)
# X, Y = np.meshgrid(_, _)
# U = -1 - X**2 + Y
# V = 1 + X - Y**2
# speed = np.sqrt(U**2, V**2)


# fig, axes = plt.subplots(2, 2, figsize=(12, 12))
# ax = axes[0][0]
# ax.streamplot(X, Y, U, V)

# ax = axes[0][1]
# ax.streamplot(X, Y, U, V, color=speed)

# ax = axes[1][0]
# lw = 6 * speed / speed.max()
# ax.streamplot(X, Y, U, V, linewidth=lw)


# ax = axes[1][1]
# lw = 6 * speed / speed.max()
# seedpoints = np.array([[0, 1], [1, 0]])
# ax.streamplot(X, Y, U, V, start_points=seedpoints)

# # print(lolo)
# plt.show()


# FUNKING WITH IMAGES
# image = plt.imread("images/vyra.png")
# plt.imshow(image)
# print(image[5].shape)


# # Animations
# def f(x, t):
#     return np.sin(x - 3 * t)


# x = np.linspace(0, 10 * np.pi, 1000)

# fig, ax = plt.subplots(1, 1, figsize=(8, 4))
# (ln1,) = plt.plot([], [])
# time_text = ax.text(
#     0.65,
#     0.95,
#     "",
#     fontsize=15,
#     transform=ax.transAxes,
#     bbox=dict(facecolor="white", edgecolor="black"),
# )

# ax.set_xlim(0, 10 * np.pi)
# ax.set_ylim(-1.5, 1.5)



# def animate(i):
#     t = 1 / 30 * i
#     # print(t, "T")
#     ln1.set_data(x, f(x, t))
#     time_text.set_text("t={:.2f}".format(i / 30))


# # print(f(x, 1 / 30 * i))

# ani = animation.FuncAnimation(fig, animate, frames=240, interval=30);
# ani.save('images/ani.gif', writer="pillow", fps=30, dpi=100)
# print(np.pi, "PI")

# ANIMATE SURFACE PLOT
_ = np.linspace(-1, 1, 100)
x, y = np.meshgrid(_, _)
z = x**2+x*y

plt.style.use(["default"])

fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
ax.plot_surface(x, y, z, cmap="coolwarm", linewidth=0, antialiased=False)
ax.view_init(elev=10, azim=0)

def animate(i):
    ax.view_init(elev=10, azim=3*i)

ani = animation.FuncAnimation(fig, animate, 120, interval=50)
ani.save("images/surface_anim.gif", "pillow", 50, 100)

plt.show()
