from . import *


class DataMeta:
    __slots__ = "pid", "batchflag", "aug"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class CachedDataset(Distributed):
    def __init__(self, loader, content: dict, meta):
        self.dics = content
        self.titles = meta["title"]
        # If meta is a dict of single item, it must be uniform for any subset of the set.
        # Otherwise meta should have multiple items. So `distribution` must be popped up here.
        self.distrib = meta.pop("distribution")
        self.loader = loader
        
        meta["shape"] = loader.load(first(content["X"])).shape
        meta["batchflag"] = hash(
            (meta["shape"], "Yb" in self.titles, "mask" in self.titles)
        )
        self.meta = meta
        Distributed.__init__(self, meta["statistic_title"])

    @staticmethod
    def hashf(*args):
        return hash(args)

    def __getitem__(self, i):
        item = {title: self.dics[title][i] for title in self.titles}
        item["meta"] = DataMeta(pid=item.pop("pid"), batchflag=self.meta["batchflag"])
        if self._fetch:
            item["X"] = self.loader.load(item["X"])
            if "mask" in item:
                item["mask"] = self.loader.load(item["mask"])

        return item

    def __len__(self):
        return len(first(self.dics.values()))

    def argwhere(self, cond, title=None, indices=None):
        if title is None:
            ge = lambda i: self[i]
        else:
            ge = lambda i: self[i][title]

        gen = range(len(self)) if indices is None else indices
        return [i for i in gen if cond(ge(i))]

    def getDistribution(self, title):
        return self.distrib[title]

    def joint(self, title1, title2):
        if title1 not in self.statTitle:
            return
        if title2 not in self.statTitle:
            return
        z = torch.zeros((self.K(title1), self.K(title2)), dtype=torch.long)

        for i in range(len(self)):
            with self.no_fetch():
                d = self[i]
            z[d[title1], d[title2]] += 1
        return z

    def K(self, title):
        return len(self.meta["classname"][title])


class CachedDatasetGroup(DistributedConcatSet):
    """
    Dataset for a group of cached datasets.
    """

    def __init__(self, path, device=None, withpid=False):
        d: dict = torch.load(os.path.join(path, "meta.pt"))
        shapedic: dict = d.pop("data")
        index: list = d.pop("index")
        # group hold the loader entity
        self.loader = IndexLoader(os.path.join(path, "images.pts"), index, device)

        datasets = [CachedDataset(self.loader, dic, d.copy()) for dic in shapedic.values()]
        DistributedConcatSet.__init__(self, datasets, tag=shapedic.keys())

