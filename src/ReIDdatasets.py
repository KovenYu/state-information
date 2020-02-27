import h5py
import numpy as np
import torch.utils.data as data
import torch
from PIL import Image
import matplotlib.pyplot as plt


class CFP(data.Dataset):
    def __init__(self, root, sets, transform=None, require_views=True):
        super(CFP, self).__init__()
        self.root = root
        self.transform = transform
        self.require_views = require_views
        if self.transform is not None:
            self.on_transform = True
        else:
            self.on_transform = False
        images, labels, views, protocol = torch.load(root)
        self.sets = sets
        self.images = []
        self.labels = []
        self.views = []
        for set in sets:
            begin = set * 500
            end = begin + 500
            self.images = self.images + images[begin:end]
            self.labels = self.labels + labels[begin:end]
            self.views = self.views + views[begin:end]
            begin_ = set * 200 + 5000
            end_ = begin_ + 200
            self.images = self.images + images[begin_:end_]
            self.labels = self.labels + labels[begin_:end_]
            self.views = self.views + views[begin_:end_]
            self.protocol = protocol[set]

    def return_protocol(self):
        return self.protocol

    def return_mean(self, axis=(0, 1, 2)):
        mean = []
        for i in axis:
            channel_mean = []
            for img in self.images:
                m = np.asarray(img)[:, :, i].mean()
                channel_mean.append(m)
            mean.append(np.asarray(channel_mean).mean())
        return np.asarray(mean)

    def return_std(self, axis=(0, 1, 2)):
        std = []
        for i in axis:
            channel_std = []
            for img in self.images:
                m = np.asarray(img)[:, :, i].std()
                channel_std.append(m)
            std.append(np.asarray(channel_std).std())
        return np.asarray(std)

    def return_num_class(self):
        return np.size(np.unique(np.asarray(self.labels)))

    def turn_on_transform(self, transform=None):
        self.on_transform = True
        if transform is not None:
            self.transform = transform
        assert self.transform is not None, 'Transform not specified.'

    def turn_off_transform(self):
        self.on_transform = False

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img, label, view = self.images[index], self.labels[index], self.views[index]

        if self.on_transform:
            img = self.transform(img)

        if self.require_views:
            return img, label, view, index
        else:
            return img, label


class MultiPie(data.Dataset):
    def __init__(self, root, state='train', transform=None, require_views=True):
        super(MultiPie, self).__init__()
        self.root = root
        self.state = state
        self.transform = transform
        self.require_views = require_views
        if self.transform is not None:
            self.on_transform = True
        else:
            self.on_transform = False
        dataset = torch.load(root)
        self.images, self.labels, self.views = dataset[state]

    def return_mean(self, axis=(0, 1, 2)):
        mean = []
        for i in axis:
            channel_mean = []
            for img in self.images:
                m = np.asarray(img)[:, :, i].mean()
                channel_mean.append(m)
            mean.append(np.asarray(channel_mean).mean())
        return np.asarray(mean)

    def return_std(self, axis=(0, 1, 2)):
        std = []
        for i in axis:
            channel_std = []
            for img in self.images:
                m = np.asarray(img)[:, :, i].std()
                channel_std.append(m)
            std.append(np.asarray(channel_std).std())
        return np.asarray(std)

    def return_num_class(self):
        return np.size(np.unique(np.asarray(self.labels)))

    def turn_on_transform(self, transform=None):
        self.on_transform = True
        if transform is not None:
            self.transform = transform
        assert self.transform is not None, 'Transform not specified.'

    def turn_off_transform(self):
        self.on_transform = False

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img, label, view = self.images[index], self.labels[index], self.views[index]

        if self.on_transform:
            img = self.transform(img)

        if self.require_views:
            return img, label, view, index
        else:
            return img, label


class Market(data.Dataset):
    def __init__(self, root, state='train', transform=None, require_views=True):
        super(Market, self).__init__()
        self.root = root
        self.state = state
        self.transform = transform
        self.require_views = require_views
        if self.transform is not None:
            self.on_transform = True
        else:
            self.on_transform = False

        f = h5py.File(self.root, 'r')
        variables = list(f.items())
        # [0]: gallery_data
        # [1]: gallery_labels
        # [2]: gallery_views
        # [3]: probe_data
        # [4]: probe_labels
        # [5]: probe_views
        # [6]: train_data
        # [7]: train_labels
        # [8]: train_views

        if self.state == 'train':
            _, temp = variables[6]
            self.data = np.transpose(temp.value, (0, 3, 2, 1))
            _, temp = variables[7]
            self.labels = np.squeeze(temp.value)
            _, temp = variables[8]
            self.views = np.squeeze(temp.value)
        elif self.state == 'gallery':
            _, temp = variables[0]
            self.data = np.transpose(temp.value, (0, 3, 2, 1))
            _, temp = variables[1]
            self.labels = np.squeeze(temp.value)
            _, temp = variables[2]
            self.views = np.squeeze(temp.value)
        elif self.state == 'probe':
            _, temp = variables[3]
            self.data = np.transpose(temp.value, (0, 3, 2, 1))
            _, temp = variables[4]
            self.labels = np.squeeze(temp.value)
            _, temp = variables[5]
            self.views = np.squeeze(temp.value)
        else:
            assert False, 'Unknown state: {}\n'.format(self.state)

    def return_mean(self, axis=(0, 1, 2)):
        return np.mean(self.data, axis)

    def return_std(self, axis=(0, 1, 2)):
        return np.std(self.data, axis)

    def return_num_class(self):
        return np.size(np.unique(self.labels))

    def turn_on_transform(self, transform=None):
        self.on_transform = True
        if transform is not None:
            self.transform = transform
        assert self.transform is not None, 'Transform not specified.'

    def turn_off_transform(self):
        self.on_transform = False

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        img, label, view = self.data[index], self.labels[index], self.views[index]

        img = Image.fromarray(img)

        if self.on_transform:
            img = self.transform(img)

        if self.require_views:
            return img, label, view, index
        else:
            return img, label


class FullTraining(data.Dataset):
    def __init__(self, root, transform=None, require_views=False):
        super(FullTraining, self).__init__()
        self.root = root
        self.transform = transform
        self.require_views = require_views
        if self.transform is not None:
            self.on_transform = True
        else:
            self.on_transform = False

        f = h5py.File(self.root, 'r')
        variables = list(f.items())
        # [0]: data
        # [1]: labels

        _, temp = variables[0]
        self.data = np.transpose(temp.value, (0, 3, 2, 1))
        _, temp = variables[1]
        self.labels = np.squeeze(temp.value)

    def return_mean(self, axis=(0, 1, 2)):
        if 'MSMT17' in self.root:
            return np.array([79.2386, 73.9793, 77.2493])
        else:
            return np.std(self.data, axis)

    def return_std(self, axis=(0, 1, 2)):
        if 'MSMT17' in self.root:
            return np.array([67.2012, 63.9191, 61.8367])
        else:
            return np.std(self.data, axis)

    def return_num_class(self):
        return np.size(np.unique(self.labels))

    def turn_on_transform(self, transform=None):
        self.on_transform = True
        if transform is not None:
            self.transform = transform
        assert self.transform is not None, 'Transform not specified.'

    def turn_off_transform(self):
        self.on_transform = False

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]

        img = Image.fromarray(img)

        if self.on_transform:
            img = self.transform(img)

        return img, label


def main():
    Market_dataset = FullTraining('data/Market.mat')
    print(Market_dataset.__len__())
    img, label = Market_dataset[0]
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    main()