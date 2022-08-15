import numpy as np
from PIL import Image
from scipy import ndimage


class NodeDetection:
    def __init__(self, img_path, N, S=1, B=10, M=150):
        self.N = N
        self.S = S
        self.B = B
        self.M = M
        self.img_path = img_path
        img = Image.open(self.img_path)
        self.rgb = np.array(img)

    def grayscale(self, pix):
        lum_weight = (0.21, 0.72, 0.07)
        return np.dot(pix[..., :3], lum_weight)

    def blur(self, pix):
        '''
        Gaussian Kernel
        '''
        size = int(self.B) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * self.S**2)
        g = np.exp(-((x**2 + y**2) / (2.0*self.S**2))) * normal
        return ndimage.convolve(pix, g)

    def sobel_operator(self, pix):
        Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        Ky = np.array([[1, 2, 1], [0, 0, -0], [-1, -2, -1]])
        Gx = ndimage.convolve(pix, Kx)
        Gy = ndimage.convolve(pix, Ky)
        G = np.hypot(Gx, Gy)
        G = G / G.max() * 255
        theta = np.arctan2(Gy, Gx)
        return (G, theta)

    def non_max_suppression(self, img, D):
        M, N = img.shape
        Z = np.zeros((M, N), dtype=np.int32)
        angle = D * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1, M-1):
            for j in range(1, N-1):
                q = 255
                r = 255
                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]
                Z[i, j] = img[i, j] if (img[i, j] >= q) and (
                    img[i, j] >= r) else 0
        return Z

    def node_detecton(self, img, E):
        kernel = np.ones([3, 3])
        nodes = ndimage.convolve(img, kernel)
        N, M = img.shape
        for i in range(N):
            for j in range(M):
                if nodes[i, j] > 9*255*E:
                    img[i, j] = 255
                else:
                    img[i, j] = 0
        return img

    def get_edges(self):
        img = Image.open(self.img_path)
        rgb = np.array(img)
        grayscale = self.grayscale(rgb)
        blur = self.blur(grayscale)
        gradient, theta = self.sobel_operator(blur)
        edges = self.non_max_suppression(gradient, theta)
        return edges

    def get_nodes(self):
        edges = self.get_edges()
        all_nodes = np.array(np.where(edges > self.M))
        rand_index = np.random.choice(
            all_nodes.shape[1], min(self.N, all_nodes.shape[1]), False)
        rand_nodes = all_nodes[:, rand_index]
        rgb_nodes = np.append(np.transpose(rand_nodes),
                              self.rgb[rand_nodes[0], rand_nodes[1]], axis=1)
        return rgb_nodes


if __name__ == '__main__':
    nodeDetection = NodeDetection("wave.jpg", N=5000, B=15)
    nodes = nodeDetection.get_nodes()
    edges = nodeDetection.get_edges()
    res_img = Image.fromarray(edges.astype('uint8'), 'L')
    res_img.show()
