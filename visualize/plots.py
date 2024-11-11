import matplotlib as mpl
import matplotlib.pyplot as plt
from optimize.derivatives import *

mpl.use('TkAgg')


def plot1D(f: Callable[[float], float], bounds: tuple[float, float], samples: int = 50) -> None:
    x1 = np.linspace(bounds[0], bounds[1], samples)
    # noinspection PyTypeChecker
    x2 = f(x1)

    plt.plot(x1, x2)
    plt.show()


def plot2D(f: Callable[[float, float], float], bounds: list[tuple[float, float]], samples: int = 50) -> None:
    x1, x2 = np.meshgrid(np.linspace(bounds[0][0], bounds[0][1], samples), np.linspace(bounds[1][0], bounds[1][1], samples))
    # noinspection PyTypeChecker
    x3 = f(x1, x2)

    plt.subplot(projection='3d').plot_surface(x1, x2, x3, cmap='plasma')
    plt.show()
