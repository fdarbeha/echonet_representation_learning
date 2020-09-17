import pathlib
import torch.utils.data
import os
import numpy as np
import collections
import skimage.draw
import tqdm
import cv2

DATA_DIR = 'PATH/TO/DATA/FOLDER'

class EchoDataset(torch.utils.data.Dataset):
    def __init__(self, root=None,
                 ssl=False,
                 split="train", target_type="EF",
                 mean=0., std=1.,
                 length=16, period=4,
                 max_length=250,
                 crops=1,
                 pad=None):
        """length = None means take all possible"""

        if root is None:
            root = DATA_DIR
            # print("DATASET ROOT: ", root + 'Videos')
        self.ssl = ssl
        self.folder = pathlib.Path(root)
        self.split = split
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.crops = crops
        self.pad = pad


        self.fnames, self.outcome = [], []
        # fnames: list of video names
        # outcome: list of entire line in excel sheet
        # frames: dictionary of {filename: frames}
        # traces: dictionary of {filename, frame: trace}

        with open(self.folder / "FileList.csv") as f:
            self.header = f.readline().strip().split(",")
            filenameIndex = self.header.index("FileName")
            splitIndex = self.header.index("Split")

            for (i, line) in enumerate(f):
                lineSplit = line.strip().split(',')

                fileName = lineSplit[filenameIndex]
                fileMode = lineSplit[splitIndex].lower()

                if (split == "all" or split == fileMode) and os.path.exists(self.folder / "Videos" / fileName):
                    self.fnames.append(fileName)
                    self.outcome.append(lineSplit) # entire line will be the outcome

            self.frames = collections.defaultdict(list) #frames is a dictionary
            self.trace = collections.defaultdict(_defaultdict_of_lists) #2D dictionary


            keep = [len(self.frames[os.path.splitext(f)[0]]) >= 2 and f != "0X4F55DC7F6080587E.avi" for f in self.fnames]
            self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
            self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]

            if ssl == False and split == 'train':
                self.fnames = self.fnames[:int(0.2 * len(self.fnames))]
                self.outcome = self.outcome[:int(0.2 * len(self.outcome))]
            

    def __getitem__(self, index):


        # print(self.fnames[index])
        video = os.path.join(self.folder, "Videos", self.fnames[index])
        video = loadvideo(video)
        # print("video shape: ", video.shape)

        # Apply normalization
        assert(type(self.mean) == type(self.std))
        if isinstance(self.mean, int) or isinstance(self.mean, float):
            video = (video - self.mean) / self.std
        else:
            video = (video - self.mean.reshape(3, 1, 1, 1)) / self.std.reshape(3, 1, 1, 1)

        c, f, h, w = video.shape #channel, frameNumber, height, weight
        if self.length is None:
            length = f // self.period
        else:
            length = self.length

        length = min(length, self.max_length)
        # print("length: ", length)

        if f < length * self.period:
            # Pad video with frames filled with zeros if too short
            video = np.concatenate((video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1)
            c, f, h, w = video.shape

        if self.crops == "all":
            start = np.arange(f - (length - 1) * self.period)
        else:
            if self.ssl == True:
                s = (f - (length - 1) * self.period)# - 25
            else:
                s = (f - (length - 1) * self.period)

            start = np.random.choice(s, self.crops)
            start2 = np.random.choice(s, self.crops)
            start3 = np.random.choice(s, self.crops)

            

        target = []
        for t in self.target_type:
            key = os.path.splitext(self.fnames[index])[0]
            target.append(np.float32(self.outcome[index][self.header.index(t)]))


        # Select random crops
        video1 = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)
        if self.ssl == True:
            try:
                video2 = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start2)
                video3 = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start3)
            
        
            except:
                start2 = [s - 5 for s in start]
                video2 = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start2)
                start3 = [s - 10 for s in start]
                video3 = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start3)
               


        if self.crops == 1:
            video = video1[0]
            if self.ssl == True:
                video2 = video2[0]
                video3 = video3[0]
            

        else:
            video = np.stack(video1)

        if self.pad is not None:
            c, l, h, w = video.shape
            temp = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            temp[:, :, self.pad:-self.pad, self.pad:-self.pad] = video
            i, j = np.random.randint(0, 2 * self.pad, 2)
            video = temp[:, :, i:(i + h), j:(j + w)]
            if self.ssl == True:
                c, l, h, w = video2.shape
                temp = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video2.dtype)
                temp[:, :, self.pad:-self.pad, self.pad:-self.pad] = video2
                i, j = np.random.randint(0, 2 * self.pad, 2)
                video2 = temp[:, :, i:(i + h), j:(j + w)]

                c, l, h, w = video3.shape
                temp = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video3.dtype)
                temp[:, :, self.pad:-self.pad, self.pad:-self.pad] = video3
                i, j = np.random.randint(0, 2 * self.pad, 2)
                video3 = temp[:, :, i:(i + h), j:(j + w)]

        
        target = 1 if target < 50 else 0
        if self.ssl == True:
            return video, video2, video3, target 

        return video, target

    def __len__(self):
        return len(self.fnames)

    def get_number_of_samples(self):
        return self.__len__()


def _defaultdict_of_lists():
    return collections.defaultdict(list)

def get_mean_and_std(dataset, split, samples=10):
    if len(dataset) > samples:
        dataset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), samples, replace=False))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)

    n = 0
    mean = 0.
    std = 0.

    split = 'Find mean and std of ' + split + 'set'
    for(i, (x, *_)) in enumerate(dataloader):
    # for (i, (x, *_)) in enumerate(tqdm.tqdm(dataloader, split)):
        x = x.transpose(0, 1).contiguous().view(3, -1)
        n += x.shape[1]
        mean += torch.sum(x, dim=1).numpy()
        std += torch.sum(x ** 2, dim=1).numpy()
    mean /= n
    std = np.sqrt(std / n - mean ** 2)

    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    return mean, std

def get_train_and_test_echonet_datasets(tasks="EF", frames=16, period=4, ssl=False):
    """
    Returns training validation and testing datasets constructed from
    echonet dataset
    """


    mean, std = get_mean_and_std(EchoDataset(split="train"), 'train')

    kwargs = {"target_type": tasks,
              "mean": mean,
              "std": std,
              "length": frames,
              "period": period,
              }
    # print(frames, period)
    dataset_train = EchoDataset(ssl=ssl, split="train", **kwargs, pad=12)

    dataset_val = EchoDataset(ssl=ssl, split="val", **kwargs, pad=12)

    dataset_test = EchoDataset(ssl=ssl, split="test", **kwargs, pad=12)

    return dataset_train, dataset_val, dataset_test

def loadvideo(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError()
    capture = cv2.VideoCapture(filename)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # empty numpy array of appropriate length, fill in when possible from front
    v = np.zeros((frame_count, frame_width, frame_height, 3), np.float32)

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        v[count] = frame

    v = v.transpose((3, 0, 1, 2))

    return v

def main():
    dataset = Echo()
    # very first line missing!!
    x, y = dataset[1]
    print(type(x), " ", x.shape)
    print(type(y), " ", y)



if __name__ == "__main__":
    main()
