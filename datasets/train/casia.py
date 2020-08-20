from torchvision.datasets import ImageFolder

class CASIAWebfaceExperiment(ImageFolder):
    def __init__(self, root, transform_a, transform_b, 
                 prior_transform=None):
        super(CASIAWebfaceExperiment, self).__init__(
            root=root,
        )
        
        self.prior_transform = prior_transform
        self.transform_a = transform_a
        self.transform_b = transform_b


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample_a, sample_b, target) where target is class_index
             of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        
        if self.prior_transform is not None:
            sample = self.prior_transform(sample)
        
        sample_a = self.transform_a(sample)
        sample_b = self.transform_b(sample)
        
        return sample_a, sample_b, target
