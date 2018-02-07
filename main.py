import numpy as np
import cv2
import sys


class SeamCarving:
    def __init__(self, img, mask, mask_weight, use_mask):
        self.h, self.w, self.channel = img.shape
        self.mask = mask
        self.mask_weight = mask_weight
        self.use_mask = use_mask

    def find_min_seam(self, out, h, w):
        seam = []
        energy = self.cal_energy(out)
        total_energy = self.cal_total_energy(energy)

        col_idx = np.argmin(total_energy[-1])
        seam_energy = total_energy[-1, col_idx]

        seam.insert(0, col_idx)
        for row in range(h - 2, -1, -1):
            if col_idx == 0:
                col_idx = np.argmin(total_energy[row, 0:min(w, col_idx + 2)])
            else:
                col_idx = np.argmin(total_energy[row, max(0, col_idx - 1):min(w, col_idx + 2)]) + col_idx - 1
            seam_energy += total_energy[row, col_idx]
            seam.insert(0, col_idx)
        return seam, seam_energy

    def delete_seam(self, h, w, out, seam, mask_opt=False):
        if mask_opt:
            out_img = np.zeros(shape=(h, w))
        else:
            out_img = np.zeros(shape=(h, w, self.channel))

        for row in range(h - 1, -1, -1):
            out_img[row] = np.delete(out[row], seam[row], axis=0)

        return out_img

    def remove_seams(self, width, height, delta, img):
        out = img.copy()
        total_cost = 0
        for idx in range(1, delta+1):
            # w indicates the width after deletion executed
            w = width - idx
            seam,cost = self.find_min_seam(out, height, w)
            out = self.delete_seam(height, w, out, seam)
            if use_mask:
                mask = np.zeros(shape=(self.mask.shape[0], self.mask.shape[1]-1))
                mask = self.delete_seam(height, w, self.mask, seam, True)
                self.mask = mask
            total_cost += cost
        return out, total_cost

    def insert_seam(self, h, w, out, seam, inserted, mask_opt=False):

        if mask_opt:
            out_img = np.zeros(shape=(h, w))
        else:
            out_img = np.zeros(shape=(h, w, self.channel))
        for row in range(h - 1, -1, -1):
            if inserted.get(row) is None:
                inserted[row] = []
            inserted[row].append(seam[row])
            addition = 2 * sum(i <= seam[row] for i in inserted[row])-1
            average = np.average(out[row, seam[row] + addition - 1:seam[row] + addition + 1], axis=0)
            out_img[row] = np.insert(out[row], seam[row] + addition, average, axis=0)
        return out_img, inserted

    def add_seams(self, delta, width, height, img):
        out = img.copy()
        remain = img.copy()
        total_cost = 0
        inserted = {}
        for idx in range(1, delta+1):
            w = width + idx
            seam, cost = self.find_min_seam(remain, height, width-idx)
            if use_mask:
                mask = np.zeros(shape=(self.mask.shape[0], self.mask.shape[1]+1))
                mask,_ = self.insert_seam(height, w, self.mask, seam, inserted, mask_opt=True)
                self.mask = mask
            remain = self.delete_seam(height, width-idx, remain, seam)
            out, inserted = self.insert_seam(height, w, out, seam, inserted)
            total_cost += cost
        return out, total_cost

    def change_seam(self, h, w, img):
        T = np.zeros(shape=(h, w))
        imgs = {}
        imgs[(0, 0)] = img

        for i in range(h):
            for j in range(w):
                if i == 0 and j > 0:
                    # find_min_seam returns the minimum energy seam and its energy
                    previous_img = imgs[(i, j-1)]
                    seam, T[i, j] = self.find_min_seam(previous_img, previous_img.shape[0], previous_img.shape[1]-1)
                    imgs[(i, j)] = self.delete_seam(previous_img.shape[0], previous_img.shape[1]-1, previous_img, seam)

                if j == 0 and i > 0:
                    rot = np.rot90(imgs[(i - 1, j)])
                    seam, T[i, j] = self.find_min_seam(rot, rot.shape[0], rot.shape[1] - 1)
                    imgs[(i, j)] = np.rot90(self.delete_seam(rot.shape[0], rot.shape[1] - 1, rot, seam), 3)

                if i > 0 and j > 0:
                    img_r = np.rot90(imgs[(i-1, j)])
                    img_c = imgs[(i, j-1)]

                    seam_r, E_r = self.find_min_seam(img_r, img_r.shape[0], img_r.shape[1]-1)
                    seam_c, E_c = self.find_min_seam(img_c, img_c.shape[0], img_c.shape[1]-1)

                    if (T[i, j-1] + E_c) < (T[i-1, j] + E_r):
                        T[i, j] = T[i, j-1] + E_c
                        imgs[(i, j)] = self.delete_seam(img_c.shape[0], img_c.shape[1]-1, img_c, seam_c)
                    else:
                        T[i, j] = T[i-1, j] + E_r
                        imgs[(i, j)] = np.rot90(self.delete_seam(img_r.shape[0], img_r.shape[1]-1, img_r, seam_r), 3)

        return imgs[(h-1, w-1)]

    def cal_energy(self, img):
        sobelx = cv2.Sobel(img, -1, 1, 0, ksize=3)
        grad_x = np.asarray([np.sum(np.absolute(sobelx)[i], 1) for i in range(len(np.absolute(sobelx)))])
        sobely = cv2.Sobel(img, -1, 0, 1, ksize=3)
        grad_y = np.asarray([np.sum(np.absolute(sobely)[i], 1) for i in range(len(np.absolute(sobely)))])

        return grad_x + grad_y

    def cal_total_energy(self, energy):
        total_energy = energy.copy()

        m, n = energy.shape

        for i in range(1, m):
            for j in range(n):
                if j == 0:
                    total_energy[i, j] = energy[i, j] + min(total_energy[i-1, j], total_energy[i-1, j+1])
                elif j == n-1:
                    total_energy[i, j] = energy[i, j] + min(total_energy[i-1, j], total_energy[i-1, j-1])
                else:
                    total_energy[i, j] = energy[i, j] + min(total_energy[i-1, j], total_energy[i-1, j+1], total_energy[i-1, j-1])
        if use_mask:
            total_energy = total_energy + self.mask * self.mask_weight
        return total_energy

    def rotate_mask90(self):
        self.mask = np.rot90(self.mask)

if __name__ == '__main__':

    arg = sys.argv

    input = arg[1]
    sal_mask = "./input.mask.jpg"

    new_h = int(arg[2])
    new_w = int(arg[3])
    mask_weight = 200

    print("Processing height: " + str(new_h)+" , width: "+ str(new_w))

    img = cv2.imread(input)

    use_mask = False
    saliency_map = img.copy()
    if cv2.imread(sal_mask) is not None:
        use_mask = True
        saliency_map = cv2.imread(sal_mask, cv2.IMREAD_GRAYSCALE)

    h, w, channel = img.shape

    del_h = new_h - h
    del_w = new_w - w

    seam_carver = SeamCarving(img, saliency_map, mask_weight, use_mask)

    if del_h < 0 and del_w < 0 :
        img = seam_carver.change_seam(del_h*-1, del_w*-1, img)
    else:
        if del_h < 0:
            seam_carver.rotate_mask90()
            rot_img = img.copy()
            rot_img = np.rot90(rot_img, 1)
            img,_ = seam_carver.remove_seams(delta=del_h*-1, width=rot_img.shape[1], height=rot_img.shape[0], img=rot_img)
            img = np.rot90(img, 3)
        if del_h > 0:
            seam_carver.rotate_mask90()
            rot_img = img.copy()
            rot_img = np.rot90(rot_img, 1)
            img,_ = seam_carver.add_seams(delta=del_h, width=rot_img.shape[1], height=rot_img.shape[0], img=rot_img)
            img = np.rot90(img, 3)
        if del_w < 0:
            img,_ = seam_carver.remove_seams(delta=del_w*-1, height=img.shape[0], width=img.shape[1], img=img)
        if del_w > 0:
            img,_ = seam_carver.add_seams(delta=del_w, height=img.shape[0], width=img.shape[1], img=img)

    cv2.imwrite('out.png', np.uint8(img))

