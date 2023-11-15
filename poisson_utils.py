import scipy.sparse as sp
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import linalg
import cv2

def solve_equation(div, dst, mask):
    '''
    Solve equation using sparse matrix
    '''
    # construct sparse matrix A
    mask_area = np.where(mask > 0)
    pixel_num = mask_area[0].size
    A = lil_matrix((pixel_num, pixel_num), dtype=np.float64)

    b = np.zeros(pixel_num, dtype=np.float64)
    hash_map = {(x, y): i for i, (x, y) in enumerate(zip(mask_area[0], mask_area[1]))}
    for i, (x, y) in enumerate(zip(mask_area[0], mask_area[1])):
        print(i)
        A[i, i] = -4
        b[i] = div[x, y]
        if (x - 1, y) in hash_map:
            A[i, hash_map[(x - 1, y)]] = 1
        elif x - 1 >= 0:
            b[i] -= dst[x - 1, y]
        if (x + 1, y) in hash_map:
            A[i, hash_map[(x + 1, y)]] = 1
        elif x + 1 < mask.shape[0]:
            b[i] -= dst[x + 1, y]
        if (x, y - 1) in hash_map:
            A[i, hash_map[(x, y - 1)]] = 1
        elif y - 1 >= 0:
            b[i] -= dst[x, y - 1]
        if (x, y + 1) in hash_map:
            A[i, hash_map[(x, y + 1)]] = 1
        elif y + 1 < mask.shape[1]:
            b[i] -= dst[x, y + 1]
    # solve equation
    A = A.tocsr()
    solution = spsolve(A, b)
    # put x into dst
    for i, (x, y) in enumerate(zip(mask_area[0], mask_area[1])):
        dst[x, y] = int(np.clip(solution[i], 0, 255))
    return dst


class Poisson:
    def __init__(self):
        pass
    def seamlessClone(self, src, dst, mask):
        dst = np.float64(dst)
        src_div = cv2.Laplacian(np.float64(src), ksize=1, ddepth=-1)
        result = [solve_equation(x, y, mask) for x, y in zip(cv2.split(src_div), cv2.split(dst))]
        return cv2.merge(result)


    def mixedClone(self, src, dst, mask):
        kernel_grad = [np.array([[0, -1, 1]]), np.array([[1, -1, 0]]), np.array([[0], [-1], [1]]),
              np.array([[1], [-1], [0]])]
        src_grad= [cv2.filter2D(np.float64(src), -1, kernel_grad[i]) for i in range(4)]
        dst_grad = [cv2.filter2D(np.float64(dst), -1, kernel_grad[i]) for i in range(4)]
        final_grad = [np.where(np.abs(src_grad[i]) >= np.abs(dst_grad[i]), src_grad[i], dst_grad[i]) for i in range(4)]
        final_grad = np.sum(final_grad, axis=0)

        result = [solve_equation(x, y, mask) for x, y in zip(cv2.split(final_grad), cv2.split(dst))]
        return cv2.merge(result)

    def textureFlatten(self, src, mask, lower, upper):
        kernel_grad = [np.array([0, -1, 1]), np.array([1, -1, 0]), np.array([[0], [-1], [1]]), np.array([[1], [-1], [0]])]
        kernel_edge = [np.array([0, 1, 1]), np.array([1, 1, 0]), np.array([[0], [1], [1]]), np.array([[1], [1], [0]])]

        canny = cv2.Canny(src, lower, upper)
        src_div = np.zeros_like(src, dtype=np.float64)
        for k1, k2 in zip(kernel_edge, kernel_grad):
            edge = cv2.filter2D(canny, -1, k1)
            grad = cv2.filter2D(np.float64(src), -1, k2)
            grad[edge == 0] = 0
            src_div += grad

        result = [solve_equation(x, y, mask) for x, y in zip(cv2.split(src_div), cv2.split(src))]
        return cv2.merge(result)


    def illuminationChange(self, src, mask, a, b):
        src_div = cv2.Laplacian(np.float64(src), ksize=1, ddepth=-1)
        new_div = src_div * (a ** b * np.log(np.linalg.norm(src_div)) ** (-b))
        result = [solve_equation(x, y, mask) for x, y in zip(cv2.split(new_div), cv2.split(src))]
        return cv2.merge(result)


    def colorChange(self, src_convert, mask, r, g, b):
        old_src = cv2.split(src_convert)
        new_src = cv2.merge((old_src[0] * r, old_src[1] * g, old_src[2] * b))
        src_div = cv2.Laplacian(new_src, ksize=1, ddepth=-1)
        result = [solve_equation(x, y, mask) for x, y in zip(cv2.split(src_div), cv2.split(src_convert))]
        return cv2.merge(result)


    def tiling(self, src):   # Dirichlet boundary condition
        src = np.float64(src)
        mask = np.ones_like(src)[:, :, 0]
        mask_area = np.where(mask > 0)
        hash_map = {(x, y): i for i, (x, y) in enumerate(zip(mask_area[0], mask_area[1]))}

        boundary_mask = np.zeros_like(mask, dtype=np.float64)
        boundary_mask[0] = 1
        boundary_mask[-1] = 1
        boundary_mask[:, 0] = 1
        boundary_mask[:, -1] = 1
        boundary_area = np.where(boundary_mask > 0)
        boundary_hash = {(x, y): i for i, (x, y) in enumerate(zip(boundary_area[0], boundary_area[1]))}

        result = []
        for n, s in enumerate(cv2.split(src)):
            src_div = cv2.Laplacian(np.float64(s), ksize=1, ddepth=-1)
            boundary_avg = np.zeros_like(s, dtype=np.float64)
            boundary_avg[0] = (s[0] + s[-1]) * 0.5
            boundary_avg[-1] = (s[0] + s[-1]) * 0.5
            boundary_avg[:, 0] = (s[:, 0] + s[:, -1]) * 0.5
            boundary_avg[:, -1] = (s[:, 0] + s[:, -1]) * 0.5

            A = lil_matrix((s.shape[0] * s.shape[1], s.shape[0] * s.shape[1]), dtype=np.float64)
            # b = np.ndarray((s.shape[0] * s.shape[1],), dtype=np.float64)
            b = np.zeros(s.shape[0] * s.shape[1], dtype=np.float64)
            for i, (x, y) in enumerate(zip(mask_area[0], mask_area[1])):
                if (x, y) in boundary_hash:
                    A[i, i] = 1
                    b[i] = boundary_avg[x, y]

                else:
                    A[i, i] = -4
                    b[i] = src_div[x, y]
                    A[i, hash_map[(x - 1, y)]] = 1
                    A[i, hash_map[(x + 1, y)]] = 1
                    A[i, hash_map[(x, y - 1)]] = 1
                    A[i, hash_map[(x, y + 1)]] = 1
            A = A.tocsr()
            solution = spsolve(A, b)

            res = np.zeros_like(s)
            for i, (x, y) in enumerate(zip(mask_area[0], mask_area[1])):
                res[x, y] = np.clip(solution[i], 0, 255)
            result.append(res)

        return cv2.merge(result)

