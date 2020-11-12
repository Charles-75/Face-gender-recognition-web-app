from PIL import Image
import matplotlib.pyplot as plt


def getwidth(path):
    img = Image.open(path)
    size = img.size
    aspect = size[0] / size[1]  # width / height
    width = 300 * aspect
    return int(width)


def get_size(path):
    img = Image.open(path)
    return img.size[0]


def gender(string):
    try:
        return string.split('_')[0].split('/')[-1]
    except:
        return None


def elbow_method(eigen_ratio, eigen_ratio_cum, n):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(eigen_ratio[:n], 'r>--')
    plt.xlabel('no. of components')
    plt.ylabel('Explained Variance ratio')
    plt.subplot(1, 2, 2)
    plt.xlabel('no. of components')
    plt.ylabel('Cumulative Explained Variance ratio')
    plt.plot(eigen_ratio_cum[:n], 'r>--')
    plt.savefig('./report/elbow_method.png')
