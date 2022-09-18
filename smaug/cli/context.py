class Context:
    def __init__(self) -> None:
        self.__transforms = []

    def register_transform(self, name: str):
        self.__transforms.append(name)

    def iter_transforms(self):
        return iter(self.__transforms)
