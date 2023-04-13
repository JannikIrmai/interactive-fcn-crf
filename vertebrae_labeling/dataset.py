import torch
from torch.utils.data import Dataset as BaseDataset
import numpy as np
from utils import generate_perturbation_matrix_3D, augment_cents, augment_im, pad_to_shape, centroids_to_mask


class TrainDataset(BaseDataset):

	def __init__(self, data_paths, sigma: float = 3):
		self.paths = data_paths
		self.sigma = sigma
		
	def __getitem__(self, i):
		# read data
		path = self.paths[i]
		data_dict = np.load(path, allow_pickle=True)
		im = data_dict['im'].astype(float)
		cents = data_dict['cents'].astype(float)

		# -- normalise
		im = im/2048.
		im[im < -1] = -1
		im[im > 1] = 1

		# -- orientation augmentation

		if np.random.random() < 0.5:
			# affine
			affine_mat = generate_perturbation_matrix_3D(max_t=10, max_s=1.25, min_s=0.75, max_r=10)

		else:
			affine_mat = np.eye(4) 
		
		h = im.shape[0]

		if np.random.random() < 0.5:
			# random crops
			h_crop_lo = np.random.randint(0, h//3)
			h_crop_hi = np.random.randint(2 * h//3, h)
		else:
			h_crop_lo = 0
			h_crop_hi = h
			
		im = augment_im(im, affine_mat, crops=(h_crop_lo, h_crop_hi))
		cents = augment_cents(cents, im.shape, affine_mat, crops=(h_crop_lo, h_crop_hi))

		# on round of centroid clean-up
		cents[cents[:, 0] > im.shape[0], :] = np.nan
		cents[cents[:, 0] < 0, :] = np.nan
		# --

		h, w, d = im.shape

		msk = centroids_to_mask(cents, (h, w, d), sigma=self.sigma)
		
		im = torch.FloatTensor(im)
		msk = torch.FloatTensor(msk)
		cents = torch.FloatTensor(cents)

		c = msk.shape[-1]

		return im, msk, cents, (h, w, d, c)

	@staticmethod
	def collate_fn(data):
		
		H, W, D = 0, 0, 0

		for item in data:
			h, w, d, C = item[-1]
			if h > H: H = h
			if w > W: W = w
			if d > D: D = d		

		im = []
		msk = []
		cents = []
		for idx, item in enumerate(data):
			
			x, pads = pad_to_shape(item[0], (H, W, D))
			im.append(x)

			x, _ = pad_to_shape(item[1], (H, W, D, C))
			msk.append(x)

			# correct cents with pad
			cents.append(item[2] + torch.tensor([pads[0], pads[2], pads[4]]))  # pads = (h_lo, h_hi, w_lo, w_hi, d_lo, d_hi)

		im = torch.stack(im, dim=0)
		msk = torch.stack(msk, dim=0)
		cents = torch.stack(cents, dim=0)

		return im, msk, cents

	def __len__(self):
		return len(self.paths)


class TestDataset(BaseDataset):

	def __init__(self, data_paths, *args, **kwargs):
		self.paths = data_paths

	def __getitem__(self, i):
		# read data
		path = self.paths[i]
		data_dict = np.load(path, allow_pickle=True)
		im = data_dict['im'].astype(float)
		cents = data_dict['cents'].astype(float)
		
		# read hr data
		try:
			path_hr = path.replace('2mm', '1mm')
			data_dict = np.load(path_hr, allow_pickle=True)
			im_hr = data_dict['im'].astype(float)
			print("loaded 1mm")
		except FileNotFoundError:
			im_hr = im
			for i in range(3):
				im_hr = np.repeat(im_hr, repeats=2, axis=i)

		# -- normalise
		im = im/2048.
		im[im < -1] = -1
		im[im > 1] = 1

		im_hr = im_hr/2048.
		im_hr[im_hr < -1] = -1
		im_hr[im_hr > 1] = 1

		return im, cents, im_hr, path

	def __len__(self):
		return len(self.paths)
