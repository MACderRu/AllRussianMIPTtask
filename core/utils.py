import numpy as np


def sc_to_angle(s, c):
    return np.round(np.rad2deg(np.arctan2(s, c))).astype(np.int32) % 360


def read_img(p, scale=4, transform=None):
    im = Image.open(p)
    linear, _ = im.size
    im = np.array(im.resize((linear // scale, linear // scale)))
    if transform is not None:
        im = transform(im).unsqueeze(0).float()
    return im


def calc_metric(y_pred, y_true):
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    
    y_pred = np.clip(y_pred, 0, 1)
    y_true = np.clip(y_true, 0, 1)
    
    xy1_pred = np.round(y_pred[:, :2] * 10496).astype(np.int32)
    xy2_pred = np.round(y_pred[:, 2:4] * 10496).astype(np.int32)
    xy_pred = (xy1_pred + xy2_pred) / 2
    angle_pred = np.rad2deg(2 * y_pred[:, 4] * np.pi)
    # angle_pred = sc_to_angle(y_pred[:, 2], y_pred[:, 3])
    
    xy1_true = np.round(y_true[:, :2] * 10496).astype(np.int32)
    xy2_true = np.round(y_true[:, 2:4] * 10496).astype(np.int32)
    xy_true = (xy1_true + xy2_true) / 2
    angle_true = np.rad2deg(2 * y_true[:, 4] * np.pi)
    # angle_true = sc_to_angle(y_true[:, 2], y_true[:, 3])
    
    cord_error = np.sum(np.abs(xy_pred - xy_true) / 10496, axis=1)
    a_err = np.abs(angle_pred - angle_true)
    angle_error = np.minimum(a_err, 360 - a_err) / 360

    return (1 - (0.7 * 0.5 * cord_error + 0.3 * angle_error)).sum()


class Cutout:
    def __init__(self, length, p, n_holes=1):
        self.n_holes = n_holes
        self.length = length
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        
        if np.random.random() > self.p:
            return img
        
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            img[:, y1: y2, x1: x2] = img[:, y1: y2, x1: x2].uniform_()

#         mask = torch.from_numpy(mask)
#         mask = mask.expand_as(img)
#         img = img * mask

        return img
